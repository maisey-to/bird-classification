NOTE: Check the requirements.txt before running
NOTE: 'cd models' before running the executables

Files to run:
	main.py
	train_resnet.py

About the executables:
    The train_resnet.py is meant to be run independently from AlexNet and SqueezeNet.
    In main.py, AlexNet and SqueezeNet will train. The argument for each function will be the number of epochs.

About the folders:
    archive: Contains the dataset CUB_200_2011
        CUB_200_2011: Contains all the images, classes, class labels, bounding boxes, and train_test_split recommendations
        segmentations: Were not used for this project
        attributes.txt: Was not used for this project
    
    models: Contains the implementations of the Dataset, AlexNet, SqueezeNet, and ResNet
        alexnet.py: Contains the implementation of AlexNet
        BirdImageDataset.py: Contains the implementation of the Dataset
        dataloader_no_boundingbox.py: Contains the implementation of the DataLoader
        main.py: Contains the executable for AlexNet and SqueezeNet
        squeezenet.py: Contains the implementation of SqueezeNet
        train_alexnet_like_resnet.py: Contains the implementation of of AlexNet with some features of the ResNet implementation
        train_alexnet: The executable for AlexNet
        train_resnet: The implementation and executable for ResNet
        train_squeezenet: The executable for SqueezeNet
    
    results:
        models: Contains the saved models
        All other text files contain the results of the training with accuracies, losses, and epochs

