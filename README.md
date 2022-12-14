# Frequency re-Scaling in Data Augmentation for Corruption-Robust Image Classification
## Abstract
Modern Convolutional Neural Networks (CNNs) are used in various artificial intelligence applications, including computer vision, speech recognition, and robotics. However, practical usage in various applications requires large-scale datasets, and real-world data contains various corruptions, which degrade the model’s performance due to inconsistencies in train and test distributions. In this study, we propose Frequency re-Scaling Data Augmentation (FSDA) to improve the classification performance, robustness against corruption, and localizability of a classifier trained on various image classification datasets. Our method is designed with two processes; a mask generation process and a pattern re-scaling process. Each process clusters spectra in the frequency domain to obtain similar frequency patterns and scale frequency by learning re-scaling parameters from frequency patterns. Since CNN classifies images by paying attention to their structural features highlighted with FSDA, CNN trained with our method has more robustness against corruption than other data augmentations (DAs). Our method achieves higher performance on three public image classification datasets such as CIFAR-10/100 and STL-10 than other DAs. In particular, our method significantly improves robustness against various corruption error by 5.47\% over baseline on average and the localizability of the classifier.

![ours](https://user-images.githubusercontent.com/117921416/201263746-1c4b54c0-0370-4768-b35f-55cc256fe88b.png)

## Experiment Results

![Screenshot from 2022-11-11 13-35-35](https://user-images.githubusercontent.com/117921416/201264189-6972b025-e983-4186-847c-1f3cb0e7019e.png)
![Screenshot from 2022-11-11 13-35-45](https://user-images.githubusercontent.com/117921416/201264257-057d1e61-ff5d-44c8-b3d1-b263b5cde0dc.png)
![Screenshot from 2022-11-11 13-35-53](https://user-images.githubusercontent.com/117921416/201264315-df8c6d7a-3d73-4598-8d52-0b773e503374.png)

## Code Usage
1. Install Pytorch 1.8 
  - Our method is implemented on Python 3.6 and torch 1.8.0

2. Clone the repository
  ```
  git clone https://github.com/cvpr2023fsda/FSDA_CVPR2023.git
  cd FSDA_CVPR2023
  ```
  
3. Dataset
  - We use three public image classification dataset(CIFAR-10/100 and STL-10).
  - For testing corruption robustness, download the [CIFAR-10-C](https://zenodo.org/record/2535967), [CIFAR-100-C](https://zenodo.org/record/3555552#.Y23TZ3VByV4) datasets. 
  - Please put datasets in folder `dataset`.

4. Evaluation
  - Download [trained model](https://drive.google.com/file/d/155-DY-H-wE4FyMRFL9i1-ireHFAJGsiS/view?usp=share_link) and put it in folder `model_save`.
  - Evaluation code is in `test.py`.
 ```
 python test.py --data_path dataset --save_path model_save --data_type [DATA_TYPE] --epochs 100 --lr 0.1 --augment [AUGMENT]
 ```
