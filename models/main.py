import BirdImageDataset
from alexnet import AlexNet
#from resnet34 import ResNet34
#from squeezenet import SqueezeNet

from training_no_boundingbox import get_train_valid_loader, get_test_loader
#import training_with_boundingbox

import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loaders
# Bird Image Dataset
train_loader, valid_loader = get_train_valid_loader(None, 
                                                    batch_size=64, 
                                                    augment=False, 
                                                    random_seed=1)
test_loader = get_test_loader(None, 
                                batch_size=64)       


def run_AlexNet():
    # Hyper-parameters
    num_classes = 200
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005

    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
            # Write results to results folder
            with open('../results/alexnet.txt', 'a') as f:
                f.write('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {}%\n'.format(epoch+1, num_epochs, loss.item(), 100 * correct / total))
    
    with open('../results/models/alexnet.pth', "w") as p:
        torch.save(model.state_dict(), p)

    with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
            # Write results to results folder
            with open('../results/alexnet.txt', 'a') as f:
                f.write('Test Accuracy: {}%\n'.format(100 * correct / total))

run_AlexNet()


# def run_ResNet34():

# def run_SqueezeNet(): 

