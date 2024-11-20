from train_alexnet import run_AlexNet
#from train_resnet import run
from train_squeezenet import run_SqueezeNet

def main():
    # Optimize hyperparameters of AlexNet using grid search
    run_AlexNet(400)
    # Optimize hyperparameters of SqueezeNet using grid search
    run_SqueezeNet(1000)

main()