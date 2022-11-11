import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

from utils import *
from dataset.Dataset import CIFAR10Dataset, CIFAR100Dataset, STL10Dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def transform_generator(args):
    test_transform_list = [
        # transforms.ToPILImage(),
        transforms.Resize(args.image_size),
        transforms.ToTensor()
    ]

    return transforms.Compose(test_transform_list)

def load_dataset(args, dataset_dir):
    if args.data_type == 'cifar10':
        if args.add_noise == 'original':
            test_data = datasets.CIFAR10(dataset_dir, train=False, transform=transform_generator(args), download=True)
        else:
            test_data = CIFAR10Dataset(args.add_noise, transform=transform_generator(args), severity=args.severity - 1)
    elif args.data_type == 'cifar100':
        if args.add_noise == 'original':
            test_data = datasets.CIFAR100(dataset_dir, train=False, transform=transform_generator(args), download=True)
        else:
            test_data = CIFAR100Dataset(args.add_noise, transform=transform_generator(args), severity=args.severity - 1)
    elif args.data_type == 'stl10':
        if args.add_noise == 'original':
            test_data = datasets.STL10(dataset_dir, split='test', transform=transform_generator(args), download=True)
        else:
            test_data = STL10Dataset(args.add_noise, transform=transform_generator(args), severity=args.severity - 1)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)

    return test_loader

def load_model(args, device) :
    model = get_model(args.num_classes)
    if torch.cuda.device_count() > 1:
        print('Multi GPU activate : ', torch.cuda.device_count())
        model = nn.DataParallel(model)
    model = model.to(device)

    model_dir = os.path.join(args.load_path, args.data_type)
    model_dirs = os.path.join(model_dir, args.augment)

    if not os.path.exists(model_dirs): os.makedirs(model_dirs)

    load_model_path = '{}_{}'.format(args.lr, str(args.epochs).zfill(3))
    load_path = os.path.join(model_dirs, '{}.pth'.format(load_model_path))

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path)
    print(".pth keys() =  {}.".format(checkpoint.keys()))

    return model

def test(args, device, model, test_loader) :
    model.eval()
    correct_top1, total = 0, 0
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(test_loader):
            if (batch_idx + 1) % 10 == 0:
                print("{}/{}({}%) COMPLETE | NOISE : {}(SEVERITY : {})".format(
                    batch_idx + 1, len(test_loader), np.round((batch_idx + 1) / len(test_loader) * 100, 4), args.add_noise, args.severity))

            image, target = image.to(device), target.to(device).long()

            output = model(image)

            total += image.size(0)

            # Calculate top 1 accuracy
            _, rank1 = torch.max(output, 1)
            correct_top1 += (rank1 == target).sum().item()

        test_top1_acc = correct_top1 / total

    return 1. - test_top1_acc

def main(args) :
    print('hello world!!')

    dataset_rootdir = os.path.join('.', args.data_path)

    try :
        dataset_dir = os.path.join(dataset_rootdir, args.data_type)
    except TypeError :
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    if 'cifar' in args.data_type :
        args.image_size = 32
        args.num_classes = 10 if args.data_type == 'cifar10' else 100
    elif 'stl' in args.data_type :
        args.image_size = 96
        args.num_classes = 10

    device = get_deivce()
    model = load_model(args, device)
    test_loader = load_dataset(args, dataset_dir)

    test_top1_err = test(args, device, model, test_loader)

    print("###################### TEST REPORT (noise = {} | severity = {}) ######################".format(args.add_noise, args.severity))
    print("test top1_err : {}".format(np.round(test_top1_err * 100, 2)))
    print("###################### TEST REPORT (noise = {} | severity = {}) ######################".format(args.add_noise, args.severity))

if __name__=='__main__' :
    import configuration
    args = argparsing()
    main(args)
    for add_noise in configuration.CORRUPTION_LIST :
        args.add_noise = add_noise
        main(args)