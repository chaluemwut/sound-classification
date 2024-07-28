import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import random_split
from classifer import AudioClassifier
from soundsd import SoundDS
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')

def process():
    download_path = '/Users/aof/Desktop/UrbanSound8K'

    # Read metadata file
    metadata_file = f'{download_path}/metadata/UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)
    df.head()

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/audio/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

    # Take relevant columns
    df = df[['relative_path', 'classID']]
    myds = SoundDS(df, download_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    num_epochs=10
    myModel = AudioClassifier()
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device
    training(myModel, train_dl, num_epochs)
    print(myModel)

if __name__ == '__main__':
    process()