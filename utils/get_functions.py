import argparse

import torch

def argparsing() :
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--data_type', type=str, default='cifar10')
    parser.add_argument('--load_path', type=str, default='model_save')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')

    # Train parameter
    parser.add_argument('--augment', type=str, default='original', help='augmenting method')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=100)

    # Test parameter
    parser.add_argument('--add_noise', type=str, default='original')
    parser.add_argument('--severity', type=int, default=1)

    # FSDA parameter
    parser.add_argument('--num_enc', type=int, default=3)
    parser.add_argument('--angle', type=int, default=10)
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--our_method_save_path', type=str, default='baseline')

    # Print parameter
    parser.add_argument('--step', type=int, default=100)

    args = parser.parse_args()

    return args

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(num_classes) :
    from models import wideresnet
    return wideresnet(num_classes, num_channels=3, depth=40, widen_factor=2)