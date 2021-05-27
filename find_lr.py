import torch.optim as optim
import torch.nn.functional as F
import wandb
import json

from torch.utils.data import DataLoader
from data_modules import MyDataset
from torch_lr_finder import LRFinder
from model import MyModel
from optimizers import define_optimizer, define_loss

# Define the model
model = MyModel()

# Load the project via json
with open("wandb_project.json") as f:
    wandb_project = json.load(f)

run = wandb.init(
    project=wandb_project["project"],
    entity=wandb_project["entity"],
    job_type="lr tuning",
)
# Define the data to load
dataloader = DataLoader(MyDataset(), batch_size=64, shuffle=True)

# Choose optimizer settings
optimizer = define_optimizer(model.parameters())  # Get the optimizer defined
criterion = define_loss()  # Loss function
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(dataloader, end_lr=10, num_iter=100)

for x, y in zip(lr_finder.history["lr"], lr_finder.history["loss"]):
    wandb.log({"lr": x, "loss": y})

run.finish()
