"""
this 
"""
import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from collections import OrderedDict
import matplotlib.pyplot as plt




# Define the HDF5Dataset class
class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        
        self.file_path = file_path # Path to the HDF5 file with the dataset
        self.transform = transform 

        self.events = [] 
        self.labels = [] 
          
        # Open the HDF5 file and get the length of the dataset
        with h5py.File(self.file_path, 'r') as file:
            self.length = len(file['charge/calib_final_hits/data'])

            interactions_events = file['mc_truth/interactions/data']['event_id'] # Need to load in 'mc_truth/interactions/data', then get 'event_id'
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
        return self.length # Return the length of the dataset

    def __getitem__(self, event):
        # Open the HDF5 file
        with h5py.File(self.file_path, 'r') as file:
            # Get the data for the event
            data = file['charge/calib_final_hits/data'][self.events[event]]
            x = data['x']  # Get the x, y, z data
            y= data['y']
            z = data['z'] 
            features = np.column_stack((x, y, z))      # Combine the x, y, z data into a (N, 3) array
            label = self.labels[event]                 # Get the label for the event

        return features, label

# To build a model, we need to:
    # 1. Setting up device 
    # 2. Define the model class by subclassing nn.Module.
    # 3. Defining a loss function and optimizer.
    # 4. Creating a training loop 

# 1. Setting up device
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#2. Model class:
    # Given 'features' as input we want the model to predict 'label' as output.
    # Creates 2 nn.Linear layers in the constructor capable of handling the input and output shapes of 'features' and 'label'.
    # Defines a forward() method containing the forward pass computation of the model.
    # Instantiates the model class and sends it to the target device.
    
class NuClassModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Args:
            input_features: Number of input features to the model.
            out_features: how many classes there are (3).
            hidden_units: Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(), # non-linear layers
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features), # how many classes are there?
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_fn(self, label_predictions, true_labels):
        loss = nn.CrossEntropyLoss()
        return loss(label_predictions, true_labels)

    def accuracy_fn(self, true_labels, label_predictions):
        correct = torch.eq(true_labels, label_predictions).sum().float().item()
        accuracy = (correct / len(label_predictions)) * 100
        return accuracy

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            total_samples = 0
            predictions=[]
            labels=[]
            self.model.train()

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                self.optimizer.zero_grad()

                outputs = self.model(features)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(labels)
                epoch_accuracy += self.accuracy_fn(labels, outputs) * len(labels)
                total_samples += len(labels)

            epoch_loss /= total_samples
            epoch_accuracy /= total_samples

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")



#Create data dataset and loaders
    
if __name__ == "__main__":
    dataset = HDF5Dataset('/home/gkufatty/Documents/research/blip-stuff/flow-files/MiniRun4_1E19_RHC.flow.00000.FLOW.h5')
    train_loader = DataLoader(dataset, batch_size=1)  # Assuming a batch size of 1 for now
    batch_size=1

    model= NuClassModel(input_features=3, 
                    output_features=3, 
                    hidden_units=8).to(device)
    model
    trainer= Trainer(model)
    trainer.train(train_loader, epochs=5)  # Adjust the number of epochs as needed



# #     # """
# #     # (1) create dataset for each file
# #     # (2) create loader for datasets
# #     # (3) create neural network
# #     # (4) create loss function
# #     # (5) create optimizer
# #     # (6) create training loop 
# #     # (7) loop over epochs, loop over files, loop over the loader
# #     # (8) save the loss after each call
# #     # (9) plot the loss v epoch
# #     # """