# Import PyTorch libraries
# Other libraries we'll use
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from CustomCNN import BasicNet
from PIL import Image

print("Libraries imported - ready to use PyTorch", torch.__version__)
class_names = [
    'Audi', 'Hyundai_Creta', 'Mahindra_Scorpio',
    'Rolls_Royce', 'Swift', 'Tata_Safari', 'Toyota_Innova'
]

batch_size = 128
epochs = 100


# Function to ingest data using training and test loaders
def load_dataset():
    # Load all the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root='kaggle/Cars_Dataset/train', transform=transform)
    test_data = datasets.ImageFolder(root='kaggle/Cars_Dataset/test', transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset()
print('Data loaders ready')

import matplotlib.pyplot as plt
import os
from random import randint


# Function to predict the class of an image
def predict_image(classifier, image_path):
    import numpy

    # Set the classifer model to evaluation mode
    classifier.eval()

    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Preprocess the image
    pil_image = Image.open(image_path)
    image_tensor = transformation(pil_image).float()

    # Add an extra batch dimension since pytorch treats all inputs as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a Variable
    input_features = Variable(image_tensor)

    # Predict the class of the image
    output = classifier(input_features)
    index = output.data.numpy().argmax()
    return index


# Create a random test image
in_image = None

# Create a new model class and load the saved weights
num_classes = len(class_names)
model = BasicNet(num_classes)
current_model = 'BasicNet'
model_file = 'model_store/cnn_car_' + current_model + '.pt'

model.load_state_dict(torch.load(model_file))

# Call the prediction function
index = predict_image(model, in_image)
print(class_names[index])
