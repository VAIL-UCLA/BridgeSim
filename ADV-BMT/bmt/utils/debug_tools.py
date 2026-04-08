import resource

from bmt.dataset.datamodule import InfgenDataModule
from bmt.utils.config import global_config, cfg_from_yaml_file
from bmt.utils.utils import REPO_ROOT

DEBUG_CONFIG_FILE = "cfgs/motion_debug.yaml"


def get_debug_config(cfg_file=DEBUG_CONFIG_FILE):
    config = cfg_from_yaml_file(REPO_ROOT / cfg_file, global_config)
    return config


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           ''' % (point, usage[0], usage[1], usage[2] / 1024.0)


def get_debug_dataloader(
    cfg_file=DEBUG_CONFIG_FILE,
    in_evaluation=True,
    config=None,
    train_batch_size=10,
    train_num_workers=0,
    val_batch_size=1,
    val_num_workers=0,
):
    if config is None:
        config = get_debug_config(cfg_file=cfg_file)
    datamodule = InfgenDataModule(
        config,
        train_batch_size=train_batch_size,
        train_num_workers=train_num_workers,
        val_batch_size=val_batch_size,
        val_num_workers=val_num_workers,
        train_prefetch_factor=2,
        val_prefetch_factor=2
    )
    datamodule.setup("fit")
    if in_evaluation:
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.train_dataloader()
    return dataloader


def get_debug_data(cfg_file=DEBUG_CONFIG_FILE, in_evaluation=True):
    dataloader = get_debug_dataloader(cfg_file, in_evaluation)
    for data in dataloader:
        return data


if __name__ == '__main__':
    data = get_debug_data()
    print(1)
