import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torch import optim


dataset = FashionMNIST(root='data/', download=True,transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])
train =  DataLoader(train, batch_size=64, shuffle=True,drop_last=True)
val = DataLoader(val, batch_size=64, shuffle=True,drop_last=True)

class SimpleRNN(nn.Module):
    """
    Just a simple RNN model for classifying the FashionMNIST dataset

    SPOILER ALERT: This model is not the best model for this dataset. 
    Literally, any other model will perform better than this one.
    """
    def __init__(self,input_size,hidden_size,output_size,num_layers = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first = False)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        h, _ = self.rnn(x) # x = (seq_len,batch_size,input_size)
        x = self.fc(h)
        return x.softmax(dim = 1) # softmax for classification

class LitSimpleRNN(pl.LightningModule):
    def __init__(self,input_size,hidden_size,output_size,num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.RNN = SimpleRNN(input_size=input_size,hidden_size=hidden_size,output_size=output_size,num_layers=num_layers)
    
    def hot_encode(self,y, num_classes=10):
        """
        One hot encode an int
        """
        y = F.one_hot(y, num_classes=10).view(10,64)
        return y.float()

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(1,0,2,3)
        x = x.reshape(1,64,28*28)
        y = self.hot_encode(y,10)
        y = y.reshape(10,1,64)
        y = y.permute(1,2,0)
        pred = self.RNN(x)
        loss = nn.CrossEntropyLoss()(pred,y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    RNN = LitSimpleRNN(28*28,64,10)
    trainer = pl.Trainer(limit_predict_batches=100,max_epochs=1)
    trainer.fit(model = RNN,train_dataloaders=train)