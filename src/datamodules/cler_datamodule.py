from datasets.cler_dataset import CLERDataset, CLERTestDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class CLERDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir, partition, batch_size=32, num_workers=1, pin_memory=False
    ):
        super(CLERDataModule, self).__init__()
        self.data_dir = data_dir
        self.partition = partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_data = self.read_list_of_lists_from_file(
            self.data_dir + f"train{self.partition}.txt"
        )
        self.test_data = self.read_list_of_lists_from_file(
            self.data_dir + f"val{self.partition}.txt"
        )

    def train_dataloader(self):
        return self._train_dataloader(
            self.train_data, self.batch_size, self.num_workers, self.pin_memory
        )

    def val_dataloader(self):
        return self._val_dataloader(
            self.test_data, self.batch_size, self.num_workers, self.pin_memory
        )

    def test_dataloader(self):
        return self._test_dataloader(
            self.test_data, self.batch_size, self.num_workers, self.pin_memory
        )

    @staticmethod
    def _train_dataloader(train, batch_size, num_workers, pin_memory):
        train_dataset = CLERDataset(train)
        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )

    @staticmethod
    def _val_dataloader(test, batch_size, num_workers, pin_memory):
        val_dataset = CLERTestDataset(test)
        return DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )

    @staticmethod
    def _test_dataloader(test, batch_size, num_workers, pin_memory):
        test_dataset = CLERTestDataset(test)
        return DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
        )

    @staticmethod
    def read_list_of_lists_from_file(filename):
        try:
            with open(filename, "r") as file:
                contents = file.readlines()
                list_of_lists = []
                for line in contents:
                    line = line.strip()
                    lst = eval(line)
                    list_of_lists.append(lst)
                return list_of_lists
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return []
        except Exception as e:
            print("An error occurred while reading the file:", str(e))
            return []
