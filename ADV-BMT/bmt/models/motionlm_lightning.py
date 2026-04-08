import logging

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from bmt.models.motionlm import MotionLM
from bmt.tokenization import get_tokenizer
from bmt.utils import lr_schedule
from bmt.utils import utils

logger = logging.getLogger(__file__)


def update_ema(target_params, source_params, rate=0.99):
    """
    PZH: From https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/engine_mar.py#L19

    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def safe_entropy(logits, epsilon=1e-5):
    """
    Computes the entropy of the given logits safely by replacing NaN and Inf values.
    :param logits: Input logits tensor.
    :param epsilon: A small value to add to the logits to avoid log(0) which results in NaN.
    :return: Mean entropy of the logits.
    """
    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)

    # Adding a small epsilon to logits to avoid log(0)
    logits = logits + epsilon

    # Compute softmax to get probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute entropy
    entropy = -(probs * torch.log(probs)).sum(-1)

    # Return the mean entropy
    return entropy.mean()


class MotionLMLightning(pl.LightningModule):
    def __init__(self, config):
        if "SEED" in config:
            pl.seed_everything(config.SEED)
            print("Everything is seeded to: ", config.SEED)
        super().__init__()
        self.config = config

        if config.MODEL.NAME in ["motionlm", "gpt"]:
            self.model = MotionLM(config=self.config)
        else:
            raise ValueError(f"Unknown model name: {config.MODEL.NAME}")

        self.save_hyperparameters(OmegaConf.to_container(self.config))

        self._tokenizer = get_tokenizer(self.config)
        # self.validation_outputs = []
        # self.validation_ground_truth = []

        self.exp_name = None

    def forward(self, batch_dict):
        return self.model(batch_dict)

    def get_loss(self, data_dict):

        loss_stat = {}
        loss = 0.0

        if self.config.USE_MOTION:

            # Get the decoder's output
            output_logit = data_dict["decoder/output_logit"]  # (B, T_skipped + 1, N, num_actions)

            # Get the GT actions
            target_action = data_dict["decoder/target_action"]  # (B, T_skipped, N)
            target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
            assert output_logit.shape[:3] == target_action.shape

            output_logit = output_logit[target_action_valid_mask]
            target_action = target_action[target_action_valid_mask]

            # Get loss
            if self.config.OPTIMIZATION.USE_FOCAL_LOSS:
                from torchvision.ops import sigmoid_focal_loss
                # Compute Focal Loss
                alpha = 0.25
                gamma = 2
                target_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                loss = sigmoid_focal_loss(
                    inputs=output_logit, targets=target_onehot, alpha=alpha, gamma=gamma, reduction="none"
                )
            else:
                loss = torch.nn.functional.cross_entropy(input=output_logit, target=target_action, reduction="none")

            original_loss = loss
            loss = loss.mean()

            assert not np.isnan(loss.item())
            assert not np.isinf(loss.item())

            with torch.no_grad():
                encodings = F.one_hot(output_logit.argmax(-1),
                                      output_logit.shape[-1]).float().reshape(-1, output_logit.shape[-1])
                avg_probs = encodings.mean(0)
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
                cluster_use = torch.sum(avg_probs > 0)

                gt_onehot = F.one_hot(target_action, output_logit.shape[-1]).float()
                gt_encodings = gt_onehot.reshape(-1, output_logit.shape[-1])
                gt_avg_probs = gt_encodings.mean(0)
                gt_perplexity = (-(gt_avg_probs * torch.log(gt_avg_probs + 1e-10)).sum()).exp()
                gt_cluster_use = torch.sum(gt_avg_probs > 0)
                debug_gt_c_use = (gt_encodings.sum(0) > 0).sum()  # .mean()

                pred_act = output_logit.argmax(-1)
                acc = torch.sum(pred_act == target_action) / target_action.shape[0]
                entropy = safe_entropy(output_logit)
                pred_act = pred_act.float()

                rate_default_pred = (pred_act == self._tokenizer.default_action).float().mean()
                rate_default_gt = (target_action == self._tokenizer.default_action).float().mean()

                num_trained_tokens = len(target_action)
                num_trained_tokens_sum = self.trainer.world_size * num_trained_tokens

                loss_stat.update(
                    {
                        "original_loss": original_loss.mean(),
                        "accuracy": acc,
                        "entropy": entropy.mean(),
                        "avg_action": pred_act.mean(),
                        "max_action": pred_act.max(),
                        "min_action": pred_act.min(),
                        "perplexity": perplexity,
                        "gt_perplexity": gt_perplexity,
                        "cluster_use": cluster_use,
                        "gt_cluster_use": gt_cluster_use,
                        "rate_84": rate_default_gt,
                        "rate_default_gt": rate_default_gt,
                        "rate_default_pred": rate_default_pred,
                        "num_trained_tokens": num_trained_tokens,
                        "num_trained_tokens_sum": num_trained_tokens_sum,
                        "toks": num_trained_tokens_sum,
                    }
                )

                if self.config.BACKWARD_PREDICTION:
                    in_back_mask = data_dict["in_backward_prediction"]
                    in_back_mask = in_back_mask.reshape(-1, 1, 1).expand(*target_action_valid_mask.shape)
                    in_back_mask = in_back_mask[target_action_valid_mask]
                    acc2 = (pred_act == target_action)
                    acc_in_back = (acc2 & in_back_mask).sum() / in_back_mask.sum()
                    acc_in_forward = (acc2 & ~in_back_mask).sum() / (~in_back_mask).sum()
                    loss_in_back = original_loss[in_back_mask].mean()
                    loss_in_forward = original_loss[~in_back_mask].mean()
                    entropy_in_back = safe_entropy(output_logit[in_back_mask]).mean()
                    entropy_in_forward = safe_entropy(output_logit[~in_back_mask]).mean()
                    loss_stat.update(
                        {
                            "accuracy_in_backward": acc_in_back,
                            "accuracy_in_forward": acc_in_forward,
                            "loss_in_backward": loss_in_back,
                            "loss_in_forward": loss_in_forward,
                            "entropy_in_backward": entropy_in_back,
                            "entropy_in_forward": entropy_in_forward,
                            "backward_ratio": in_back_mask.float().mean(),
                        }
                    )

        if self.config.RECONSTRUCT_MAP:
            gt_map_feat = data_dict["encoder/map_feature"]
            map_feat_valid_mask = data_dict["encoder/map_valid_mask"]
            polypoint_valid_mask = data_dict["encoder/map_feature_valid_mask"]
            polypoint_valid_mask = polypoint_valid_mask[map_feat_valid_mask]  # (valid points, 128)
            map_feat = gt_map_feat[map_feat_valid_mask]  # (num_valid_map_features, 128, 27)
            polypoint = map_feat[:, :, :2]  # (valid map feat, 128, 2)
            num_points = polypoint.shape[1]
            gt_valid_mask = polypoint_valid_mask.unsqueeze(-1).expand_as(polypoint)
            gt = torch.where(gt_valid_mask, polypoint, torch.zeros_like(polypoint))
            gt_valid_mask = gt_valid_mask.reshape(-1, num_points * 2)
            gt = gt.reshape(-1, num_points * 2)
            map_token = data_dict["encoder/map_token"]
            out = self.model.map_recon_head(self.model.map_recon_head_prenorm(map_token[map_feat_valid_mask]))

            # out.shape = (num_valid_map_features, 128 * 2)
            map_recon_loss = torch.nn.functional.mse_loss(out, gt, reduction="none")
            map_recon_loss = map_recon_loss[gt_valid_mask]
            map_recon_loss = map_recon_loss.mean()

            loss += map_recon_loss
            loss_stat["map_recon_loss"] = map_recon_loss
            loss_stat["map_recon_mask_rate"] = gt_valid_mask.float().mean()

        # DEBUG CODE to find unused parameters:
        # gs = torch.autograd.grad(loss.mean(), self.parameters(), allow_unused=True, retain_graph=True)
        # ns = [n for n, v in self.named_parameters()]
        # printed = False
        # for c, g in enumerate(gs):
        #     if g is None:
        #         print(ns[c])
        #         printed = True
        # if not printed:
        #     print("No unused parameters found.")


        loss_stat["total_loss"] = loss
        try:
            loss_stat["lr"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        except RuntimeError:
            # When debugging, the model might not be attached to a trainer.
            pass
        return loss, loss_stat

    def training_step(self, data_dict, batch_idx):

        # For profiling GPU usage.
        # torch.cuda.empty_cache()

        data_dict = self(data_dict)

        loss, loss_stat = self.get_loss(data_dict)

        pbar_keys = ("total_loss", "toks", "lr")

        motion_stat = {k: v for k, v in loss_stat.items() if k.startswith("motion_stat")}
        loss_stat = {k: v for k, v in loss_stat.items() if not k.startswith("motion_stat")}

        self.log_dict(
            {f"{k}": float(v)
             for k, v in loss_stat.items() if k in pbar_keys},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            prog_bar=True,
        )
        if motion_stat:
            self.log_dict(
                {f"{k}": float(v)
                 for k, v in motion_stat.items()},
                batch_size=data_dict["encoder/map_feature"].shape[0],
                prog_bar=False,
            )
        self.log_dict(
            {f"train/{k}": float(v)
             for k, v in loss_stat.items()},
            batch_size=data_dict["encoder/map_feature"].shape[0],
            # on_epoch=True,
            prog_bar=False,
        )
        self.log('monitoring_step', float(self.global_step))
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def validation_step(self, data_dict, batch_idx):

        if self.config.EVAL_MOTION:

            if data_dict["encoder/map_valid_mask"].shape[1] == 0:
                sid = data_dict["scenario_id"]
                print("Warning: Empty map_valid_mask found for scenario: ", sid)
                logger.error(f"Empty map_valid_mask found for scenario: {sid}")
                return

            try:
                self.evaluator.validation_step(
                    data_dict,
                    batch_idx,
                    model=self.model,
                    global_rank=self.global_rank,
                    trainer=self.trainer,
                    logger=self.logger,
                    log_func=self.log,
                    log_dict_func=self.log_dict,
                    print_func=self.print,
                    lightning_model=self,
                )
            except Exception as error:
                scenario_ids = data_dict["scenario_id"]
                rank = self.global_rank
                msg = f"Error in validation_step: {batch_idx=}, {scenario_ids=}, {rank=}, {error=}"
                print(msg)
                raise RuntimeError(msg) from error

    def on_validation_epoch_end(self):
        """
        This function gathers intermediate evaluation result and pass them to the Waymo
        evaluation pipeline together and log the final results.
        """
        if self.config.EVAL_MOTION:
            self.log("monitoring_step", float(self.global_step))
            self.evaluator.on_validation_epoch_end(
                global_rank=self.global_rank,
                trainer=self.trainer,
                logger=self.logger,
                log_func=self.log,
                log_dict_func=self.log_dict,
                print_func=self.print,
                exp_name=self.exp_name,
            )

    def configure_optimizers(self):
        """Required by Lightning."""
        opt_cfg = self.config.OPTIMIZATION

        if opt_cfg.OPTIMIZER == 'Adam':
            # optimizer = torch.optim.Adam(
            #     [each[1] for each in self.named_parameters()],
            #     lr=opt_cfg.LR,
            #     weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            # )
            raise ValueError()
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=opt_cfg.LR,
                weight_decay=opt_cfg.get('WEIGHT_DECAY', 0),
                betas=(0.9, 0.95),
                eps=1e-5
            )
        else:
            assert False

        if opt_cfg.get('SCHEDULER', None) == 'cosine':

            utils.rank_zero_print("=====================================")
            if self.trainer.train_dataloader is not None:
                num_steps_per_epoch = len(self.trainer.train_dataloader)
            elif self.trainer.datamodule is not None and self.trainer.datamodule.train_dataset is not None:
                utils.rank_zero_print(
                    "Finding num_steps_per_epoch from datamodule...", len(self.trainer.datamodule.train_dataset),
                    self.trainer.datamodule.train_batch_size, self.trainer.world_size
                )
                num_steps_per_epoch = len(self.trainer.datamodule.train_dataset
                                          ) // (self.trainer.datamodule.train_batch_size * self.trainer.world_size)
            else:
                raise ValueError("Can't find num_steps_per_epoch")

            num_epochs = self.config.epochs
            total_steps = num_steps_per_epoch * num_epochs
            utils.rank_zero_print("Configuring cosine scheduler")
            utils.rank_zero_print("Num Steps per epoch: ", num_steps_per_epoch)
            utils.rank_zero_print("Num Epochs: ", num_epochs)
            utils.rank_zero_print("Total Steps: ", total_steps)
            utils.rank_zero_print("=====================================")

            scheduler = lr_schedule.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            raise ValueError()
            # def lr_lbmd(cur_epoch):
            #     cur_decay = 1
            #     for decay_step in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20]):
            #         if cur_epoch >= decay_step:
            #             cur_decay = cur_decay * opt_cfg.LR_DECAY
            #     return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)
            #
            # scheduler = LambdaLR(optimizer, lr_lbmd)

        elif opt_cfg.get('SCHEDULER', None) == 'linear':
            raise ValueError()
            scheduler = lr_schedule.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
                num_training_steps=opt_cfg.TRAINING_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        elif opt_cfg.get('SCHEDULER', None) == 'inverse_sqrt':
            scheduler = lr_schedule.get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=opt_cfg.WARMUP_STEPS,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

        else:
            raise ValueError()
