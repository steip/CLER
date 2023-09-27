import torch
from torch.utils.data import Dataset


class CLERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        user_id = row[0]
        item_id = row[1]
        exp_id = row[2]
        all_exp = row[4]

        return {
            "user_id": user_id,
            "item_id": item_id,
            "exp_id": exp_id,
            "all_exps": all_exp,
        }

    def collate_fn(self, batch):
        user_ids, item_ids, exp_ids, all_exps = [], [], [], []
        for entry in batch:
            user_ids.append(entry["user_id"])
            item_ids.append(entry["item_id"])
            exp_ids.append(entry["exp_id"])
            all_exps.append(entry["all_exps"])

        return (
            torch.tensor(user_ids),
            torch.tensor(item_ids),
            torch.tensor(exp_ids),
            all_exps,
        )


class CLERTestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        user_id = row[0]
        item_id = row[1]
        all_exp = row[2]

        return {"user_id": user_id, "item_id": item_id, "all_exps": all_exp}

    def collate_fn(self, batch):
        user_ids, item_ids, all_exps = [], [], []
        for entry in batch:
            user_ids.append(entry["user_id"])
            item_ids.append(entry["item_id"])
            all_exps.append(entry["all_exps"])

        return torch.tensor(user_ids), torch.tensor(item_ids), all_exps
