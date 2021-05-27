# Run the tests
import torch
from model import MyModel
from data_modules import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the model
model = MyModel()
state_dict = torch.load('model.pt')
model.load_state_dict(state_dict)

# Set into evaluation
model.eval()
model.to('cuda')

# Define the test dataset
test_data = DataLoader(MyDataset(), batch_size = 64, shuffle = False)

with torch.no_grad():
    # Evaluate the data

    # Feed into the data
    for batch in tqdm(iter(test_data),total = len(test_data)):
        # Do stuff
        pass

    # Aggregate the data

    # Process the outputs 


