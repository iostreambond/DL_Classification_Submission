# Import PyTorch libraries
# Other libraries we'll use
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from CustomCNN import ImprovedNet, ImprovedNetLite, BasicNet

print("Libraries imported - ready to use PyTorch", torch.__version__)
class_names = [
    'Audi', 'Hyundai_Creta', 'Mahindra_Scorpio',
    'Rolls_Royce', 'Swift', 'Tata_Safari', 'Toyota_Innova'
]

epochs = 150


# Function to ingest data using training and test loaders
def load_dataset(in_batch_size):
    # Load all the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root='kaggle/Cars_Dataset/train', transform=transform)
    test_data = datasets.ImageFolder(root='kaggle/Cars_Dataset/test', transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=in_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=in_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print("Epoch:", epoch)

    # Fetch the learning rate from the optimizer
    learning_rate = optimizer.param_groups[0]['lr']

    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # back propagate
        loss.backward()
        optimizer.step()

        # Calculate the number of correct predictions
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

        # Calculate the total number of samples
        total += target.size(0)

        # Print metrics for every 10 batches so we see some progress
        if batch_idx % 10 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))

    # Calculate average training accuracy
    avg_accuracy = 100. * correct / total
    print('Training set: Average accuracy: {:.2f}%'.format(avg_accuracy))

    return avg_loss, avg_accuracy, learning_rate


def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't back propagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss, 100. * correct / len(test_loader.dataset)


# Now use the train and test functions to train and test the model

device = "cpu"
if torch.cuda.is_available():
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)

model_list = ['BasicNet', 'ImprovedNet', 'ImprovedNetLite']
num_classes = len(class_names)
patience = 10

batch_size_list = [512, 448, 384, 320, 256]

for batch_size_value in batch_size_list:

    # Get the iterative dataloaders for test and training data
    train_loader, test_loader = load_dataset(batch_size_value)
    print('Data loaders ready')

    os.makedirs('model_store_' + str(batch_size_value) + '/', exist_ok=True)
    os.makedirs('plot_' + str(batch_size_value) + '/', exist_ok=True)

    for current_model in model_list:

        print("=============================")
        print(current_model)
        print("=============================")

        if current_model == 'Net':
            model = BasicNet(num_classes=num_classes).to(device)
        elif current_model == 'ImprovedNet':
            model = ImprovedNet(num_classes=num_classes).to(device)
        else:
            model = ImprovedNetLite(num_classes=num_classes).to(device)

        # Use an "Adam" optimizer to adjust weights
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Specify the loss criteria
        loss_criteria = nn.CrossEntropyLoss()

        # Track metrics in these arrays
        epoch_nums = list()
        training_loss = list()
        validation_loss = list()
        training_accuracy = list()
        validation_accuracy = list()
        learning_rate = list()
        best_loss = np.inf
        no_improvement_count = 0
        for epoch in range(0, epochs):
            train_loss, train_accuracy, learning_r = train(model, device, train_loader, optimizer, epoch)
            test_loss, test_accuracy = test(model, device, test_loader)
            epoch_nums.append(epoch)
            training_loss.append(train_loss)
            validation_loss.append(test_loss)
            training_accuracy.append(train_accuracy)
            validation_accuracy.append(test_accuracy)
            learning_rate.append(learning_r)

            # Implementing Early Stopping
            if train_loss < best_loss:
                best_loss = train_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check if validation loss has not improved for 'patience' epochs
            if no_improvement_count >= patience:
                print(f'Early stopping: no improvement for {patience} epochs')
                break

        print(current_model)
        print("training_loss -- validation_loss")
        print(epoch_nums)
        print(training_loss)
        print(validation_loss)

        plot1 = plt.figure(1)
        plt.plot(epoch_nums, training_loss)
        plt.plot(epoch_nums, validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.title("Training and Validation Loss")
        plt.savefig('plot_' + str(batch_size_value) + '/CustomCNN_train_test_loss_' + current_model + '.png')
        plt.clf()

        print(current_model)
        print("training_accuracy -- validation_accuracy")
        print(epoch_nums)
        print(training_accuracy)
        print(validation_accuracy)
        plot2 = plt.figure(2)
        plt.plot(epoch_nums, training_accuracy)
        plt.plot(epoch_nums, validation_accuracy)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.title("Training and Validation Accuracy")
        plt.savefig('plot_' + str(batch_size_value) + '/CustomCNN_train_test_accuracy_' + current_model + '.png')
        plt.clf()

        # Save the model weights
        model_file = 'model_store_' + str(batch_size_value) + '/cnn_car_' + current_model + '.pt'
        torch.save(model.state_dict(), model_file)
        print('model saved as', model_file)

        # Pytorch doesn't have a built-in confusion matrix metric, so we'll use SciKit-Learn
        from sklearn.metrics import confusion_matrix

        # Set the model to evaluate mode
        model.eval()

        # Get predictions for the test data and convert to numpy arrays for use with SciKit-Learn
        print("Getting predictions from test set...")
        true_labels = list()
        predictions = list()
        for data, target in test_loader:
            for label in target.cpu().data.numpy():
                true_labels.append(label)
            for prediction in model.cpu()(data).data.numpy().argmax(1):
                predictions.append(prediction)

        # Plot the confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plot3 = plt.figure(3)
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.xlabel("Predicted Shape")
        plt.ylabel("Actual Shape")
        plt.title("Confusion Matrix")

        # Add annotations with actual values
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.savefig('plot_' + str(batch_size_value) + '/CustomCNN_confusion_' + current_model + '.png')
        plt.clf()

        print(current_model)
        print(cm)
