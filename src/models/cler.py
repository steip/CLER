import pytorch_lightning as pl
from models.cler_model import PytorchCLER
import torch.optim as optim
from utils.utils import calculate
import torch


class CLERLightningModule(pl.LightningModule):
    def __init__(
        self,
        user_dim,
        item_dim,
        exp_dim,
        dimension,
        dropout_rate,
        mu,
        temperature,
        topk,
        device,
        lr,
    ):
        super(CLERLightningModule, self).__init__()
        self.model = PytorchCLER(
            user_dim=user_dim,
            item_dim=item_dim,
            exp_dim=exp_dim,
            dimension=dimension,
            dropout_rate=dropout_rate,
            mu=mu,
            temperature=temperature,
            topk=topk,
            device=device,
        )
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {pytorch_total_params}")
        self.lr = lr
        self.dev = device
        # Accumulated metrics
        self.ndcg_total = 0
        self.pre_total = 0
        self.rec_total = 0
        self.count_total = 0

    def forward(self, user_ids, item_ids, exp_ids=None, pos_indices=None):
        return self.model(user_ids, item_ids, exp_ids, pos_indices)

    def training_step(self, batch, batch_idx):
        user_ids, item_ids, exp_ids, all_exps = batch
        pos_indices = []
        for idx, _exps in enumerate(all_exps):
            for jdx, _exp in enumerate(_exps):
                pos = torch.where(exp_ids == _exp)[0]
                if pos.shape[0] > 0:
                    pos_indices.append((idx, pos[0]))
        loss = self.forward(
            user_ids, item_ids, exp_ids, torch.tensor(pos_indices).to(self.dev)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, item_ids, all_exps = batch
        outputs = self.forward(user_ids, item_ids)
        ndcg, pre, rec, count = calculate(all_exps, outputs, 10)
        self.ndcg_total += ndcg
        self.pre_total += pre
        self.rec_total += rec
        self.count_total += count

    def on_validation_epoch_end(self):
        precision = self.pre_total / self.count_total
        recall = self.rec_total / self.count_total
        self.log("val_ndcg", self.ndcg_total / self.count_total)
        self.log("val_pre", precision)
        self.log("val_rec", recall)
        self.log("val_f1", 2 * precision * recall / (precision + recall + 1e-10))
        self.log("val_count", self.count_total)
        # Reset accumulators for the next validation epoch
        self.ndcg_total = 0
        self.pre_total = 0
        self.rec_total = 0
        self.count_total = 0

    def test_step(self, batch, batch_idx):
        user_ids, item_ids, all_exps = batch
        outputs = self.forward(user_ids, item_ids)
        ndcg, pre, rec, count = calculate(all_exps, outputs, 10)
        self.ndcg_total += ndcg
        self.pre_total += pre
        self.rec_total += rec
        self.count_total += count

    def on_test_epoch_end(self):
        precision = self.pre_total / self.count_total
        recall = self.rec_total / self.count_total
        self.log("test_ndcg", self.ndcg_total / self.count_total)
        self.log("test_pre", precision)
        self.log("test_rec", recall)
        self.log("test_f1", 2 * precision * recall / (precision + recall + 1e-10))
        self.log("test_count", self.count_total)
        self.ndcg_total = 0
        self.pre_total = 0
        self.rec_total = 0
        self.count_total = 0

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
