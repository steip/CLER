import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from codecarbon import track_emissions

from datamodules.cler_datamodule import CLERDataModule
from models.cler import CLERLightningModule


# @track_emissions(project_name="CLER")
def main(args):
    # Set up seed for reproducibility
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")  # high
    # Create data module
    data_module = CLERDataModule(
        data_dir=args.data_dir,  # os.path.join("../data/", args.data_dir),
        partition=args.partition,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    data_module.setup()
    # Check if the data is loaded correctly
    # for batch in data_module.train_dataloader():
    #     print(batch)
    #     break

    # Create the LightningModule
    model = CLERLightningModule(
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        exp_dim=args.exp_dim,
        dimension=args.dimension,
        dropout_rate=args.dropout_rate,
        mu=args.mu,
        temperature=args.temperature,
        topk=args.topk,
        device=args.acelerator if args.acelerator == "cpu" else "cuda:0",
        lr=args.lr,
    )

    # Set up Trainer and train the model
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.acelerator,
        precision=args.precision,
        enable_progress_bar=args.enable_progress_bar,
    )
    trainer.fit(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add arguments as needed, for example:
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--user_dim", type=int, required=True)
    parser.add_argument("--item_dim", type=int, required=True)
    parser.add_argument("--exp_dim", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--dropout_rate", type=float, default=0.88)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument(
        "--acelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--enable_progress_bar", type=bool, default=True)

    args = parser.parse_args()
    main(args)
