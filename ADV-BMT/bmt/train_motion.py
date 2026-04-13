import datetime
import os
import pathlib

import hydra
import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import OmegaConf

import bmt.utils as utils
from bmt.dataset.datamodule import InfgenDataModule
from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils import REPO_ROOT, get_time_str

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
def main(config):
    # Unfreeze the config to allow modification
    OmegaConf.set_struct(config, False)
    config.ROOT_DIR = REPO_ROOT
    OmegaConf.set_struct(config, True)

    from bmt.utils.config import global_config, cfg_from_yaml_file
    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs/motion_default.yaml", global_config)

    pl.seed_everything(config.seed)
    print("Everything is seeded to: ", config.seed)

    # Set up config
    # cfg_file = REPO_ROOT / config.cfg_file
    # config = cfg_from_yaml_file(cfg_file, global_config)
    exp_name = config.exp_name
    max_epochs = config.epochs  #or config.OPTIMIZATION.NUM_EPOCHS
    batch_size = config.batch_size
    val_batch_size = config.val_batch_size
    num_workers = config.num_workers
    val_num_workers = config.val_num_workers
    log_dir = config.log_dir or None
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)

    # Setup wandb logger
    trial_id = get_time_str(no_time=True)
    name = "{}_{}".format(exp_name, trial_id)
    if log_dir:
        save_dir = pathlib.Path(log_dir / "lightning_logs")
    else:
        save_dir = pathlib.Path(os.path.join(REPO_ROOT, "lightning_logs"))
    if config.wandb and not config.eval:
        with open(os.path.abspath(os.path.expanduser("~/wandb_api_key_file.txt")), "rt") as fp:
            api_key = fp.readline().strip()
        wandb.login(key=api_key)
        logger = WandbLogger(
            name=name,
            save_dir=save_dir,
            id=name,
            project="infgen",
            log_model=False,
            group=exp_name,
        )
    else:
        logger = TensorBoardLogger(save_dir=save_dir / "infgen", name=name)

    ckpt_save_dir = pathlib.Path(save_dir).absolute() / "infgen" / name

    # Set up trainer arguments
    callbacks = [
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            monitor="monitoring_step",
            every_n_epochs=1,
            save_last=True,
            auto_insert_metric_name=True,
            mode="max",
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            filename=str(name) + "_{epoch}-{step}",
            train_time_interval=datetime.timedelta(minutes=30),
            auto_insert_metric_name=True,
            save_on_train_epoch_end=True,
            every_n_train_steps=None,
            every_n_epochs=None,
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    device = "auto" if torch.cuda.is_available() else "cpu"
    trainer_kwargs = dict(
        num_sanity_val_steps=config.num_sanity_val_steps,
        limit_val_batches=config.limit_val_batches if config.limit_val_batches >= 0 else None,
        limit_train_batches=config.limit_train_batches if config.limit_train_batches >= 0 else None,
        gradient_clip_val=config.OPTIMIZATION.GRAD_NORM_CLIP,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=device,
        devices="auto",
        log_every_n_steps=2,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        check_val_every_n_epoch=config.get("check_val_every_n_epoch", 1),
        # strategy='ddp_find_unused_parameters_true'
    )

    # from lightning.pytorch.profilers import PyTorchProfiler
    # profiler = PyTorchProfiler(filename="profile")
    # trainer_kwargs.update(
    #     profiler=profiler,
    # )

    # if config.debug:
    #     # from lightning.pytorch.profilers import PyTorchProfiler
    #     # profiler = PyTorchProfiler(filename="profile")
    #     trainer_kwconfig.update(
    #         num_sanity_val_steps=0,
    #         # profiler=profiler,
    #         detect_anomaly=True,
    #         limit_val_batches=2,
    #         limit_train_batches=2,
    #         log_every_n_steps=1,
    #     )
    #     num_workers = 0
    #     val_num_workers = 0
    # if bf16:
    #     trainer_kwargs["precision"] = "bf16-mixed"

    datamodule = InfgenDataModule(
        config,
        train_batch_size=batch_size,
        train_num_workers=num_workers,
        train_prefetch_factor=config.prefetch_factor,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        val_prefetch_factor=config.prefetch_factor,
    )
    if torch.cuda.device_count() > 1:
        trainer_kwargs["strategy"] = 'ddp'
        # trainer_kwargs["strategy"] = 'ddp_find_unused_parameters_true'
    if log_dir:
        trainer_kwargs["default_root_dir"] = log_dir

    # Set up trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Set up model
    ckpt_path = config.ckpt
    if ckpt_path is not None:
        ckpt_path = REPO_ROOT / pathlib.Path(ckpt_path).expanduser()
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.ckpt"
        ckpt_path = str(ckpt_path.resolve().absolute())
        assert os.path.isfile(ckpt_path), ckpt_path
        assert ckpt_path.endswith(".ckpt"), ckpt_path
        print("==============================")
        print("Loading checkpoint: ", ckpt_path)
        print("==============================")

    pretrained_path = config.pretrain
    if pretrained_path:
        pretrained_path = pathlib.Path(pretrained_path).expanduser()
        pretrained_path = REPO_ROOT / pretrained_path
        if pretrained_path.is_dir():
            pretrained_path = pretrained_path / "last.ckpt"
        pretrained_path = str(pretrained_path.absolute().resolve())
        assert os.path.isfile(pretrained_path), pretrained_path
        assert pretrained_path.endswith(".ckpt"), pretrained_path
        print("==============================")
        print("Loading pretrained model: ", pretrained_path)
        print("==============================")

        map_location = None
        if not torch.cuda.is_available():
            print("CUDA is not available. Loading model on CPU!")
            print("CUDA is not available. Loading model on CPU!")
            print("CUDA is not available. Loading model on CPU!")
            map_location = "cpu"

        model = utils.load_from_checkpoint(
            checkpoint_path=pretrained_path,
            cls=MotionLMLightning,
            config=config,
            default_config=default_config,
            strict=True,
            checkpoint_surgery_func=utils.checkpoint_surgery_func,
            map_location=map_location
        )
        # model = MotionLMLightning.load_from_checkpoint(checkpoint_path=pretrained_path, strict=strict, **config)
    else:
        model = MotionLMLightning(config=config)
    model.exp_name = name

    assert model.config == config, "The config system is not working properly! Original:\n{}\n\nNew:\n{}".format(
        model.config, config
    )
    config_save_path = ckpt_save_dir / "config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    utils.rank_zero_print(summarize(model, max_depth=5))
    utils.rank_zero_print("==============================")
    utils.rank_zero_print("Root Directory: ", save_dir / "infgen")
    utils.rank_zero_print("Checkpoint Log Directory: ", ckpt_save_dir)
    utils.rank_zero_print("Config Save Path: ", config_save_path)
    utils.rank_zero_print("Exp Group: ", name)
    utils.rank_zero_print("Exp Full Name: ", name)
    utils.rank_zero_print("==============================")
    print("Rank {} is done setting up the model.".format(trainer.global_rank))
    OmegaConf.save(config, config_save_path)

    if config.eval:
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
