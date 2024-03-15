import os
import torch
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import random_split


# Import tqdm for progress bar
from tqdm.auto import tqdm

from timeit import default_timer as timer 

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Defining the HDF5Dataset class
class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        self.events = []
        self.labels = []

        with h5py.File(self.file_path, 'r') as file:
            self.length = len(file['charge/calib_final_hits/data'])

            interactions_events = file['mc_truth/interactions/data']['event_id']
            isCC = file['mc_truth/interactions/data']['isCC']
            nu_pdg = file['mc_truth/interactions/data']['nu_pdg']
            segments_events = file['mc_truth/segments/data']['event_id']
            charge_segments = file['mc_truth/calib_final_hit_backtrack/data']['segment_id']
            segments_ids = file['mc_truth/segments/data']['segment_id']
            
            for ii, event_id in enumerate(np.unique(interactions_events)):
                if isCC[ii]:
                    # CCnu_e
                    if abs(nu_pdg[ii]) == 12:
                        self.labels.append(0)
                    # CCnu_mu
                    elif abs(nu_pdg[ii]) == 14:
                        self.labels.append(1)
                    else:
                        continue
                else:
                    self.labels.append(2)
                self.events.append(
                    np.any(
                        np.isin(
                            charge_segments, segments_ids[(segments_events == event_id)]
                        ),
                        axis=1,
                    )
                )

    def __len__(self):
        return self.length

    def __getitem__(self, event):
        with h5py.File(self.file_path, 'r') as file:
            data_index = self.events[event]  
            data = file['charge/calib_final_hits/data'][data_index]
            #features = data['x', 'y', 'z']
            features = np.stack([data['x'], data['y'], data['z']], axis=1)  # Combining x, y, z into a single array
            label = self.labels[event]
            #convert to tensor
            features = torch.tensor(features, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.long)
        return features, label


hdf5_dataset = HDF5Dataset('/home/gkufatty/Documents/research/blip-stuff/flow-files/MiniRun4_1E19_RHC.flow.00000.FLOW.h5')  # Correct path to your HDF5 file
# check the dataset, dim of features is (N,3)

total_size = len(hdf5_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
# Split the dataset
train_dataset, test_dataset = random_split(hdf5_dataset, [train_size, test_size])

# Create data loaders
# Setup the batch size hyperparameter
BATCH_SIZE = 32

train_dataloader = DataLoader(train_dataset, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)
test_dataloader = DataLoader(test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)


class TestModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)

# Need to setup model with input parameters
model_0 = TestModel(input_shape=3, 
    hidden_units=10, # how many units in the hiden layer
    output_shape=3).to(device) # one for every class

print(model_0) 


# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs 
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0

    # Add a loop to loop through training batches
    for features, labels in train_dataloader:
        model_0.train() 
        # 1. Forward pass
        y_pred = model_0(features)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, labels)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(features)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)
           
            # 2. Calculate loss 
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# # Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))
