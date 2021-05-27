import wandb
import argparse
import pytorch_lightning as pl
import json
import torch
from data_modules import MyDataModule
from pytorch_lightning.loggers import WandbLogger
from model import MyModel
from optimizers import define_optimizer, define_scheduler


def train(args):
    with open("wandb_project.json") as f:
        wandb_project = json.load(f)

    # Start wandb logging
    run = wandb.init(
        project=wandb_project["project"],
        entity=wandb_project["entity"],
        job_type="training",
    )

    wandb_logger = WandbLogger()
    
    # Define the datamodule
    data_module = MyDataModule()

    # Define the model and optimizers
    model = MyModel()
    model.optimizer = define_optimizer(model.parameters())  # Set the optimizer
    model.scheduler = define_scheduler()  # Set the scheduler

    # Define the trainer arguments
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=-1, accelerator="dp", logger = wandb_logger)

    # Run
    trainer.fit(model, data_module)

    # Save the state dictionary only
    torch.save(model.state_dict(), "model.pt")

    run.finish()


# Save the model
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Training epochs", type=int, default=5)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
