import os
import random
import time
import numpy as np
import torch
from efficientnet.model import EfficientNet
from torchsummary import summary

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# tensorboardX
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


model_name = 'efficientnet-b0'
BATCH_SIZE = 16
SEED = 44
EPOCHS = 4
lr_rate = 0.001
img_size = 32  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

train_dataset = MNIST(".", train=True, transform=transform, download=True)
test_dataset = MNIST(".", train=False, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)



model = EfficientNet.from_name(model_name)

image_size = EfficientNet.get_image_size(model_name)        
print(f'\n\nalthough b0 input img size is {image_size}; you can get away with a resize of {img_size}. \n')

# adjust the final linear layer in and out features.
feature = model._fc.in_features
model._fc = torch.nn.Linear(feature,10)

model.to(device)
summary(model, (1, img_size, img_size))


# ------------------------------------------

torch.manual_seed(SEED)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

# train the model
total_step = len(train_loader)
model.train()
start_time = time.time()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)
        writer.add_scalar("Loss/train", loss, epoch)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

train_time = time.time() - start_time
print('*'*25)
print(f'\n\n Time taken to train the model :: {train_time/60:.2f} minutes.')
print('*'*25)

writer.flush()

# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

PATH = "mnist.pt"
torch.save(model, PATH)
