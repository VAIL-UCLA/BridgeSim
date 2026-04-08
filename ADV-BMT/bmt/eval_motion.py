import argparse
import datetime
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from bmt.dataset.datamodule import InfgenDataModule
from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils import global_config, cfg_from_list, cfg_from_yaml_file

# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")  # Enable TF32 matrix multiplication

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--num_sanity_val_steps', type=int, default=20, required=False, help='batch size for training')
    parser.add_argument('--limit_val_batches', type=int, default=-1, required=False, help='batch size for training')
    parser.add_argument('--limit_train_batches', type=int, default=-1, required=False, help='batch size for training')
    parser.add_argument('--train_prefetch_factor', type=int, default=2, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    # parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    # parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--debug', action='store_true', default=False, help='')
    parser.add_argument('--eval', action='store_true', default=False, help='')
    parser.add_argument('--precision', type=str, default=None, help='precision')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    # parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument(
        '--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed'
    )

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')

    # PZH: added
    parser.add_argument('--wandb', action='store_true', default=False, help='')

    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    # parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, global_config)
    global_config.TAG = Path(args.cfg_file).stem
    global_config.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, global_config)

    return args, global_config


def main():
    args, cfg = parse_config()

    exp_name = args.exp_name
    max_epochs = args.epochs  #or cfg.OPTIMIZATION.NUM_EPOCHS
    batch_size = args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    val_batch_size = args.val_batch_size or 2
    num_workers = args.num_workers
    precision = args.precision
    if precision in ["16", 16, "bf16", "bf16-mixed"]:
        print("Setting torch.set_float32_matmul_precision('medium') because you are using half precision.")
        torch.set_float32_matmul_precision("medium")
    else:
        print("Do not set torch.set_float32_matmul_precision since you are using full precision.", precision)

    model = MotionLMLightning(config=cfg)

    # Setup wandb logger
    trial_id = get_time_str()
    name = "{}_{}".format(exp_name, trial_id)
    if args.wandb:
        with open(os.path.abspath(os.path.expanduser("~/wandb_api_key_file.txt")), "rt") as fp:
            api_key = fp.readline().strip()
        wandb.login(key=api_key)
        save_dir = os.path.join(REPO_ROOT, "lightning_logs")
        logger = WandbLogger(name=name, save_dir=save_dir, id=name, project="infgen", log_model=True, group=exp_name)
    else:
        save_dir = os.path.join(REPO_ROOT, "lightning_logs")
        logger = TensorBoardLogger(save_dir=save_dir, name=exp_name)

    trainer_kwargs = dict(

        # Debug only:
        # num_sanity_val_steps=2,
        # max_epochs=max_epochs,
        # profiler=profiler,
        # detect_anomaly=True,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches > 0 else None,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches > 0 else None,
        gradient_clip_val=cfg.OPTIMIZATION.GRAD_NORM_CLIP,
        max_epochs=max_epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=2,

        # strategy='ddp_find_unused_parameters_true'
    )

    if args.eval:
        trainer_kwargs["limit_train_batches"] = 1
        trainer_kwargs["num_sanity_val_steps"] = 0

    if args.debug:
        datamodule = InfgenDataModule(
            cfg.DATA_CONFIG,
            train_batch_size=batch_size,
            train_num_workers=0,
            train_prefetch_factor=None,
            val_batch_size=2,
            val_num_workers=0,
            val_prefetch_factor=None
        )

    else:
        datamodule = InfgenDataModule(
            cfg.DATA_CONFIG,
            train_batch_size=batch_size,
            train_num_workers=num_workers,
            train_prefetch_factor=args.train_prefetch_factor,
            val_batch_size=val_batch_size,
            val_num_workers=4,
            val_prefetch_factor=1,
        )

    # if args.debug:
    # trainer_kwargs["limit_train_batches"] = 100
    # trainer_kwargs["limit_val_batches"] = 100
    # trainer_kwargs["max_epochs"] = 2

    if precision is not None:
        trainer_kwargs["precision"] = precision

    if torch.cuda.device_count() > 1:
        trainer_kwargs["strategy"] = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = args.ckpt
    if ckpt_path is not None:
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
        assert os.path.isfile(ckpt_path)
        assert ckpt_path.endswith(".ckpt")
        print("==============================")
        print("Loading checkpoint: ", ckpt_path)
        print("==============================")

    trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
