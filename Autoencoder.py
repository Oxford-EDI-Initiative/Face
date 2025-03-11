import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import glob

#building the actual autencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(128*128*3, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 128*128*3),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class CNN_Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Encoder: Convolutional layers + MaxPooling
        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # input channels = 3 (RGB), output channels = 32
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsampling by 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsampling by 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsampling by 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling by 2
            nn.Flatten(),
            nn.Linear(8*8*256, latent_dim)
        )

        self.encoder_linear_layers = nn.Sequential(
            nn.Linear(8*8*256, latent_dim)
        )

        # Decoder: Convolutional Transpose (Upsampling)
        self.decoder_linear_layers = nn.Sequential(
            nn.Linear(latent_dim, 8*8*256)
        )

        self.decoder_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder_conv_layers(x)

    def decode(self, z):
        z = self.decoder_linear_layers(z).view(-1, 256, 8, 8)
        z = self.decoder_conv_layers(z)
        return z
        
    def forward(self, x):
        encoded = self.encode(x)  
        decoded = self.decode(encoded)     
        return encoded, decoded

#defining functions to train the autoencoder
def get_batch(data, batch_size=64):
    num_imgs = data.shape[0]
    for i in range(0, num_imgs, batch_size):
        yield data[i:min(i+batch_size, num_imgs)]

def save_model(model, epoch, losses):
    path = f'autoencoder_weights\\model_weights{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'training_loss': losses[0],
        'validation_loss': losses[1]
    }, path)
    print('Model saved')

def train_epoch(X_train, model, loss, optimizer, batch_size, is_cnn):
    #keeping track of the loss across all batches
    total_loss = 0.0

    for batch in get_batch(X_train, batch_size):
        #reshaping for a FC layer if not using CNNs
        if not is_cnn:
            inputs = batch.view(-1, 128*128*3)
        else:
            inputs = batch.permute(0,3,1,2)
        
        #zeroing the gradient to prevent any propagation of gradients between steps
        optimizer.zero_grad()
        #making a pass through the model
        encoded, decoded = model(inputs)

        #reshaping the outputs to align with the inputs if not using CNNs
        if not is_cnn:
            outputs = decoded.view(-1, 128*128*3)
        else:
            #shifting the dimensions to put the channel last instead of channel first (pytorch default)
            outputs = decoded

        #computing the loss and updating weights using backprop
        batch_loss = loss(inputs, outputs)
        batch_loss.backward()
        optimizer.step()

        #updating the total loss
        total_loss += batch_loss

    return total_loss

def eval_epoch(X_val, model, loss, is_cnn):
    #keeping track of the total loss
    total_loss = 0.0

    #setting the model to evaluation mode to disable dropout and batch normalisation
    model.eval()

    #computing the loss for each batch
    for batch in get_batch(X_val):
        #flattening the input if not using a CNN
        if not is_cnn:
            inputs = batch.view(-1, 128*128*3)
        else:
            inputs = batch.permute(0,3,1,2)
        
        
        #disables computing the gradient to speed up the process
        with torch.set_grad_enabled(False):
            #forward pass through the network
            encoded, decoded = model(inputs)

            #flatenning the output if not using a CNN
            if not is_cnn:
                outputs = decoded.view(-1, 128*128*3)
            else:
                #shifting the dimensions to put the channel last instead of channel first (pytorch default)
                outputs = decoded.permute(0,2,3,1)

            #updating the total loss with the current batch loss
            current_loss = loss(inputs, outputs)
            total_loss += current_loss

    return total_loss

def train(X_train, X_val, model, num_epochs=10, batch_size=32, is_cnn=False):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(X_train, model, loss, optimizer, batch_size, is_cnn)
        val_loss = eval_epoch(X_val, model, loss, is_cnn) 
        history.append((train_loss, val_loss))
        print(f'Epoch {epoch}, train_loss {train_loss}, val_loss {val_loss}')
        save_model(model, epoch, history[epoch])

    return history

#training the autoencoder
if __name__ == "__main__":
    dataset = []

    #loading the data into a dataframe
    genders_df = pd.read_csv('lfw_attributes.txt', sep='\t', skiprows=1)['Male']
    dataset = []

    #loading the data into a dataframe
    for i, path in enumerate(glob.glob('Resized-Labelled-Faces-Wild\\*jpg')):
        print(i)
        img = plt.imread(path)
        if genders_df.at[i] > 0:
            gender = 0.9999
        else:
            gender = 0.0001
        dataset.append({'image':img, 'gender':gender})
        if i == 10000:
            break

    #splitting the data into training and validation sets
    dataset_df = pd.DataFrame(dataset)
    X = np.stack(dataset_df.image.to_numpy()) / 255.0
    y = dataset_df.gender.to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    #converting to pytorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)

    #creating and training the model
    autoencoder = Autoencoder(latent_dim=200)
    history = train(X_train, X_val, autoencoder)

    """
    for male_path in glob.glob('Resized-Male-Female\\Male\\*.jpg'):
        img = plt.imread(male_path)
        dataset.append({'image':img, 'gender':0})
    print('Male Loaded')

    for female_path in glob.glob('Resized-Male-Female\\Female\\*.jpg'):
        img = plt.imread(female_path)
        dataset.append({'image':img, 'gender':1})
    print('Female Loaded')
    """