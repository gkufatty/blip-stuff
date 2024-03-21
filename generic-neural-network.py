import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
from helper_functions import accuracy_fn 


# Import tqdm for progress bar
from tqdm.auto import tqdm
from timeit import default_timer as timer 

# def print_train_time(start: float, end: float, device: torch.device = None):
#     total_time = end - start
#     print(f"Train time on {device}: {total_time:.3f} seconds")
#     return total_time

# Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class_names = ['CCnu_e', 'CCnu_mu', 'NC']

#--------------------------------------------------(1) Getting Datasets Ready--------------------------------------------------



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
        print(f"Total events: {len(self.events)}, Total labels: {len(self.labels)}")
        with h5py.File(self.file_path, 'r') as file: 
            data = file['charge/calib_final_hits/data'][self.events[event]]
            features = np.stack([data['x'], data['y'], data['z']], axis=1)  # Combining x, y, z into a single array
            label = self.labels[event]
        #convert to tensor
        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

#------------------------------------------CNN NETWORK------------------------------------------
    
# Create a convolutional neural network 
class RegularCNNModel(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


#--------------------- General Structure


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device:torch.device=device):
    train_loss = 0
    model.to(device)
    model.train()
    for batch, (in_features,true_labels) in enumerate(data_loader):
        in_features,true_labels = in_features.to(device), true_labels.to(device)
        pred_labels = model(in_features)
        loss = loss_fn(pred_labels, true_labels)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} %")        



def eval_model(model: torch.nn.Module,  
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device:torch.device=device):
    """Returns a dictionary containing the results of model predicting on data_loader.
    """
    loss=0
    model.eval()
    with torch.inference_mode():
        for in_features, true_labels in data_loader:
            in_features,true_labels = in_features.to(device), true_labels.to(device)
            pred_labels = model(in_features)
            loss += loss_fn(pred_labels, true_labels)
        loss /= len(data_loader)
    return {'model_name': model.__class__.__name__,
            'model_loss': loss.item()}

#-----------------------------Training loop-----------------------------

# Create data loaders
BATCH_SIZE = 32 # how many samples per batch?
train_dataset = HDF5Dataset('/home/gkufatty/Documents/research/blip-stuff/flow-files/MiniRun4_1E19_RHC.flow.00000.FLOW.h5')
train_dataloader = DataLoader(train_dataset, # dataset to turn into iterable
                              batch_size=BATCH_SIZE, # how many samples per batch? 
                              shuffle=True # shuffle data every epoch?
                              )
# Create model
model= RegularCNNModel(input_shape=3, 
                       hidden_units=10, 
                       output_shape=len(class_names)).to(device)
#set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=0.1)


# Train model
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
model_results = eval_model(
    model=model,
    data_loader=train_dataloader,
    loss_fn=loss_fn
)

print(model_results)

# class Trainer:
#     def __init__(self,
#         model
#     ):
#         self.model = model
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
#         self.loss_fn = nn.CrossEntropyLoss()


#     def train(self, dataloader, num_epochs):
#         for epoch in range(num_epochs):
#             self.model.train()
#             train_loss = 0.0
#             all_predictions = []
#             all_labels = []

#             for batch_input, batch_label in dataloader:
#                 pred_output = self.model(batch_input)
#                 loss=self.loss_fn(pred_label, batch_label)
#                 train_loss += loss
#                 optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#         train_loss /= len(dataloader)
#         print(f"Train loss: {train_loss:.5f}")      
            

#     def infer(self, dataloader):
#         self.model.eval()
#         input = []
#         target = []
#         for batch_input, batch_target in dataloader:
            
#         return 

# if __name__ == "__main__":
#     dataset = HitDataset([
        
#     ])
#     batch_size = 1

#     # Create data loaders.
#     train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
#     #model = PPNModel(num_input_features=1, num_output_features=4)
#     model = PPNWithUNet(1, 4)
#     trainer = HitProposalTrainer(model)
#     trainer.train(train_dataloader, 50)

    


# train_dataset = HDF5Dataset('/home/gkufatty/Documents/research/blip-stuff/flow-files/MiniRun4_1E19_RHC.flow.00000.FLOW.h5') 

# feat1,label1=hdf5_dataset[0]
# train_size = int(0.8 * len(hdf5_dataset))
# test_size = len(hdf5_dataset)- train_size
# # Split the dataset
# train_dataset, test_dataset = random_split(hdf5_dataset, [train_size, test_size])
# print(f"Total size of dataset: {len(hdf5_dataset)}. Training set: {len(train_dataset)}. Testing set: {len(test_dataset)}")
# print(f'Features shape is (N,3): {feat1.shape}')





#------------------training and testing steps------------------



    

# def test_step(data_loader: torch.utils.data.DataLoader,
#               model: torch.nn.Module,
#               loss_fn: torch.nn.Module,
#               accuracy_fn,
#               device:torch.device=device):
#     test_loss, test_acc = 0,0
#     model.to(device)
#     model.eval()
#     with torch.inference_mode():
#         for X,y in data_loader:
#             X,y= X.to(device),  y.to(device)
#             test_pred = model(X)
#             loss = loss_fn(test_pred, y)
#             test_loss+=loss
#             test_acc+=accuracy_fn(y_true=y, 
#                                   y_pred=test_pred.argmax(dim=1))   
#         test_loss /= len(data_loader)
#         test_acc /= len(data_loader)    
#         print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


# #------------------baseline model------------------


# class BaselineModel(nn.Module):
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Flatten(), # neural networks like their inputs in vector form
#             nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
#             nn.Linear(in_features=hidden_units, out_features=output_shape)
#         )
    
#     def forward(self, x):
#         return self.layer_stack(x)

# torch.manual_seed(42)
# model_1 = BaselineModel(input_shape=3, # number of input features
#     hidden_units=10,
#     output_shape=len(class_names)).to(device) # send model to GPU if it's available
# print(model_1)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model_1.parameters(), 
#                             lr=0.1)
# # Train and test model 
# epochs = 3
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n---------")
#     train_step(data_loader=train_dataloader, 
#         model=model_1, 
#         loss_fn=loss_fn,
#         optimizer=optimizer,
#         accuracy_fn=accuracy_fn,
#         device=device
#     )
#     test_step(data_loader=test_dataloader,
#         model=model_1,
#         loss_fn=loss_fn,
#         accuracy_fn=accuracy_fn,
#         device=device
#     )

# # class TestModel(nn.Module):
# #     def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
# #         super().__init__()
# #         self.layer_stack = nn.Sequential(
# #             nn.Flatten(), 
# #             nn.Linear(in_features=input_shape, out_features=hidden_units), 
# #             nn.Linear(in_features=hidden_units, out_features=output_shape)
# #         )
    
# #     def forward(self, x):
# #         return self.layer_stack(x)
    
# # torch.manual_seed(42)

# # # Need to setup model with input parameters
# # model_0 = TestModel(input_shape=3, 
# #     hidden_units=10, # how many units in the hiden layer
# #     output_shape=3).to(device) # one for every class

# # print(model_0) 


# # # Setup loss function and optimizer
# # loss_fn = nn.CrossEntropyLoss()
# # optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# # # Set the seed and start the timer
# # torch.manual_seed(42)
# # train_time_start_on_cpu = timer()

# # # Set the number of epochs 
# # epochs = 3

# # # Create training and testing loop
# # for epoch in tqdm(range(epochs)):
# #     print(f"Epoch: {epoch}\n-------")
# #     ### Training
# #     train_loss = 0

# #     # Add a loop to loop through training batches
# #     for features, labels in train_dataloader:
# #         model_0.train() 
# #         # 1. Forward pass
# #         y_pred = model_0(features)

# #         # 2. Calculate loss (per batch)
# #         loss = loss_fn(y_pred, labels)
# #         train_loss += loss # accumulatively add up the loss per epoch 

# #         # 3. Optimizer zero grad
# #         optimizer.zero_grad()

# #         # 4. Loss backward
# #         loss.backward()

# #         # 5. Optimizer step
# #         optimizer.step()

# #         # Print out how many samples have been seen
# #         if batch % 400 == 0:
# #             print(f"Looked at {batch * len(features)}/{len(train_dataloader.dataset)} samples")

# #     # Divide total train loss by length of train dataloader (average loss per batch per epoch)
# #     train_loss /= len(train_dataloader)
    
# #     ### Testing
# #     # Setup variables for accumulatively adding up loss and accuracy 
# #     test_loss, test_acc = 0, 0 
# #     model_0.eval()
# #     with torch.inference_mode():
# #         for X, y in test_dataloader:
# #             # 1. Forward pass
# #             test_pred = model_0(X)
           
# #             # 2. Calculate loss 
# #             test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

# #             # 3. Calculate accuracy (preds need to be same as y_true)
# #             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
# #         # Calculations on test metrics need to happen inside torch.inference_mode()
# #         # Divide total test loss by length of test dataloader (per batch)
# #         test_loss /= len(test_dataloader)

# #         # Divide total accuracy by length of test dataloader (per batch)
# #         test_acc /= len(test_dataloader)

# #     ## Print out what's happening
# #     print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# # # # Calculate training time      
# # train_time_end_on_cpu = timer()
# # total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
# #                                            end=train_time_end_on_cpu,
# #                                            device=str(next(model_0.parameters()).device))
