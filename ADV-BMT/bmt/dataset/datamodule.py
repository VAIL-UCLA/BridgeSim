"""
This is a wrapper to wrap our dataset as a lightning datamodule.
"""
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from bmt.dataset import dataset


class InfgenDataModule(pl.LightningDataModule):
    def __init__(
        self, config, train_batch_size, train_num_workers, train_prefetch_factor, val_batch_size, val_num_workers,
        val_prefetch_factor
    ):
        super().__init__()
        self.config = config
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.val_prefetch_factor = val_prefetch_factor

    def setup(self, stage: str):
        self.train_dataset = dataset.InfgenDataset(config=self.config, mode="training")
        self.val_dataset = dataset.InfgenDataset(config=self.config, mode="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=self.train_num_workers,
            shuffle=True,
            persistent_workers=True if self.train_num_workers > 0 else False,
            collate_fn=self.train_dataset.collate_batch,
            prefetch_factor=self.train_prefetch_factor if self.train_num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            pin_memory=True,
            num_workers=self.val_num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate_batch,
            prefetch_factor=self.val_prefetch_factor if self.val_num_workers > 0 else None,
        )
