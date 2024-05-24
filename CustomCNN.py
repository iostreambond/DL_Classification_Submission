# Import PyTorch libraries
# Other libraries we'll use
import torch.nn as nn
import torch.nn.functional as F


# Create a neural net class
class BasicNet(nn.Module):
    # Constructor
    def __init__(self, total_classes):
        super(BasicNet, self).__init__()

        # Our images are RGB, so input channels = 3. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A second convolutional layer takes 12 input channels, and generates 12 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A third convolutional layer takes 12 inputs and generates 24 outputs
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # Our 128 x 128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # So our feature tensors are now 32 x 32, and we've generated 24 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to  the probability for each class
        self.fc = nn.Linear(in_features=56 * 56 * 24, out_features=total_classes)

    def forward(self, x):
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
        # print("After conv1 and pool:", x.shape)

        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        # print("After conv2 and pool:", x.shape)

        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(self.conv3(x)))
        # print("After conv3 and drop:", x.shape)

        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)

        # Flatten
        x = x.view(-1, 56 * 56 * 24)
        # print("After flattening:", x.shape)

        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return log_softmax tensor
        return F.log_softmax(x, dim=1)


class ImprovedNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Apply third convolutional layer followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the feature maps
        x = x.view(-1, 64 * 28 * 28)

        # Apply dropout
        x = self.dropout(x)

        # Apply first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Apply second fully connected layer
        x = self.fc2(x)

        # Return log probabilities
        return F.log_softmax(x, dim=1)


class ImprovedNetLite(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedNetLite, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU activation, batch normalization, and pooling
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))

        # Apply second convolutional layer followed by ReLU activation, batch normalization, and pooling
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))

        # Apply third convolutional layer followed by ReLU activation, batch normalization, and pooling
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))

        # Apply fourth convolutional layer followed by ReLU activation, batch normalization, and pooling
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))

        # Flatten the feature maps
        x = x.view(-1, 64 * 14 * 14)

        # Apply dropout
        x = self.dropout(x)

        # Apply first fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Apply second fully connected layer
        x = self.fc2(x)

        # Return log probabilities
        return F.log_softmax(x, dim=1)
