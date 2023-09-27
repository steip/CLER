import torch.nn as nn
import torch


class PytorchCLER(nn.Module):
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
    ):
        super(PytorchCLER, self).__init__()
        self.user_embeddings = nn.Embedding(user_dim, dimension)
        self.item_embeddings = nn.Embedding(item_dim, dimension)
        self.exp_embeddings = nn.Embedding(exp_dim, dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.mu = mu
        self.temperature = temperature
        self.topk = topk
        self.device = device
        self.init_weights()

    def init_weights(self):
        initrange = 0.005
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.exp_embeddings.weight.data.uniform_(-initrange, initrange)

    def nt_bxent_loss(self, matrix, pos_indices):
        """
        NT-BXent Loss
        :param matrix: cosine similarity matrix
        :param pos_indices: list of positive indices for each row (user-item pair)
        :param temperature: temperature parameter
        """
        pos_indices = torch.cat(
            [
                pos_indices,
                torch.arange(matrix.size(0))
                .reshape(matrix.size(0), 1)
                .expand(-1, 2)
                .to(self.device),
            ],
            dim=0,
        )

        # Ground truth labels
        target = torch.zeros(matrix.size(0), matrix.size(1)).to(self.device)
        target[pos_indices[:, 0], pos_indices[:, 1]] = 1.0
        loss = torch.nn.LogSoftmax(dim=0)(
            matrix / self.temperature
        ) + torch.nn.LogSoftmax(dim=1)(matrix / self.temperature)
        target_pos = target.bool()
        target_neg = ~target_pos
        loss_pos = torch.zeros(
            matrix.size(0), matrix.size(0), device=self.device
        ).masked_scatter(target_pos, loss[target_pos])
        loss_neg = torch.zeros(
            matrix.size(0), matrix.size(0), device=self.device
        ).masked_scatter(target_neg, loss[target_neg])
        loss_pos = loss_pos.sum(dim=1)
        loss_neg = loss_neg.sum(dim=1)

        num_pos = target.sum(dim=1)
        num_neg = matrix.size(0) - num_pos
        return -((loss_pos / num_pos) + (loss_neg / num_neg)).mean()

    def forward(self, user_ids, item_ids, exp_ids, pos_indices):
        if exp_ids is not None:
            user_emb = self.dropout(self.user_embeddings(user_ids))
            item_emb = self.dropout(self.item_embeddings(item_ids))
            exp_emb = torch.nn.functional.normalize(self.exp_embeddings(exp_ids))
            ui = torch.nn.functional.normalize(
                user_emb * self.mu + item_emb * (1 - self.mu)
            )
            logits = ui @ exp_emb.t()
            loss = self.nt_bxent_loss(logits, pos_indices)
            return loss
        else:
            user_emb = self.user_embeddings(user_ids)
            item_emb = self.item_embeddings(item_ids)
            exp_emb = torch.nn.functional.normalize(self.exp_embeddings.weight)
            ui = torch.nn.functional.normalize(
                user_emb * self.mu + item_emb * (1 - self.mu)
            )
            logits = ui @ exp_emb.t()
            return torch.topk(logits, self.topk, 1)[1]
