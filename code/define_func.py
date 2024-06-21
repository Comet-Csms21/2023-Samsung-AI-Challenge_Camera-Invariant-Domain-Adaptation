import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
#from torchsummary import summary
from torchinfo import summary

from tqdm import tqdm
from itertools import zip_longest
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
import monai
import monai.losses
import monai.metrics

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# 클래스 수
num_classes = 13 # 12개의 class와 1개의 background

# 각 클래스에 대한 색상 정의 (임의의 값)
colors = np.array([[68, 1, 84], #0: Road
                   [72, 30, 112], #1: Sidewalk
                   [68, 57, 130], #2: Construction
                   [58, 82, 139], #3: Fence
                   [48, 103, 141], #4: Pole
                   [40, 123, 142], #5: Traffic Light
                   [32, 144, 140], #6: Traffic Sign
                   [32, 164, 133], #7: Nature
                   [53, 183, 120], #8: Sky
                   [94, 201, 97], #9: Person
                   [144, 214, 67], #10: Rider
                   [199, 224, 31], #11: Car
                   [253, 231, 36], #12: Background
                   ])

# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def visualize_combined_masks(rle_data_list, shape):
    combined_mask = np.full(shape, 12, dtype=np.uint8)
    i = 0
    for rle_data in rle_data_list:
        mask = rle_decode(rle_data, shape)
        #np.savetxt(f"./output{i}.txt", mask, fmt="%d")
        combined_mask[mask == 1] = i
        i += 1
    #np.savetxt("./combine.txt", combined_mask, fmt="%d")
    if np.any(combined_mask > 12):
        print("ERROR")
    #visualize(combined_mask=combined_mask)
    return combined_mask

def grayscale_to_seeable(image):
    # 클래스를 볼 수 있게 만들기
    seeable_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(num_classes - 1):
        # 이미지에서 픽셀 값이 i인 부분의 인덱스 찾기
        indices = np.where(np.all(image == [i, i, i], axis=-1))

        # 추출된 픽셀을 모두 i*19로 바꿔줍니다.
        seeable_mask[indices] = [i*19, i*19, i*19]

    indices = np.where(np.all(image == [12, 12, 12], axis=-1))
    seeable_mask[indices] = [255, 255, 255]

    return seeable_mask

def grayscale_to_rgb(image):
    """
    Convert grayscale image to RGB
    """
    # 클래스에 해당하는 색상으로 변환
    colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for class_index in range(num_classes):
        colored_mask[image == class_index] = colors[class_index]
    if np.any(image == 255):
        colored_mask[image == 255] = colors[12]
    return colored_mask

def blend_images(image1, image2, alpha):
    """
    Blend two images using alpha transparency
    """
    blended_image = alpha * image1 + (1 - alpha) * image2
    return blended_image.astype(np.uint8)

def visualize_with_transparency(image, mask, alpha):
    """
    Visualize two images with transparency
    """
    # Convert grayscale image to RGB if necessary
    if len(mask.shape) == 2:  # If mask is grayscale
        mask = grayscale_to_rgb(mask)
    
    blended_image = blend_images(image, mask, alpha)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blended_image)
    plt.title("Image-Mask")
    plt.axis('off')
    
    plt.show()

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.data.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
def combined_masks(rle_data_list, shape):
    combined_mask = np.full(shape, 12, dtype=np.uint8)
    i = 0
    for rle_data in rle_data_list:
        mask = rle_decode(rle_data, shape)
        combined_mask[mask == 1] = i
        i += 1
    if np.any(combined_mask > 12):
        print("ERROR")
    return combined_mask

transform = A.Compose(
    [   
        A.Resize(224, 384),
        A.Normalize(),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [   
        A.Resize(540, 960),
        #A.Normalize(),
        ToTensorV2()
    ], is_check_shapes=False
)

check_transform = A.Compose(
    [   
        A.Resize(540, 960),
        #A.Normalize(),
        #ToTensorV2()
    ], is_check_shapes=False
)

def class_wieghts(dataset):
    pixel_list = []

    for i in range(13): # class 개수만큼 반복
        count = 0
        for j in range(len(dataset)): # dataset 개수만큼 반복
            # i인 요소를 True로, 아닌 경우를 False로 하는 마스크 생성
            mask = (dataset[j][1] == i)

            # True인 요소의 개수를 세어 i의 개수 계산
            count += torch.sum(mask).item()

        pixel_list.append(count)

    # mask class별 개수
    print("각 원소 개수:", pixel_list)

    # pixel_list 총합 계산
    total = sum(pixel_list) # 224 * 384 * len(dataset)

    # 각 원소의 비율을 계산하여 새로운 리스트 생성
    ratios = [(x / total) * 100 for x in pixel_list]

    #print("각 원소의 비율:", ratios)

    # 각 비율을 소숫점 둘째 자리까지 표현
    ratios_rounded = [round(x, 2) for x in ratios]

    print("각 원소의 비율(소수점 둘째 자리까지):", ratios_rounded)

    # 각 원소 비율의 역수
    ratios_weights = [round(1 / x, 2) for x in ratios_rounded]

    print("각 원소의 가중치(소수점 둘째 자리까지):", ratios_weights)

    # ratios_weight 총합 계산
    total2 = sum(ratios_weights)

    ratios_weights_norm = [x / total2 for x in ratios_weights] # 총합을 1로 만들기 위함

    print("각 원소의 가중치(총합 1):", ratios_weights_norm)

    total3 = sum(ratios_weights_norm)

    print(total3)

    return ratios_weights

def delete_folders_with_keyword(root_dir, keyword):
    """
    Delete folders that contain a specific keyword in their name.
    
    :param root_dir: The root directory to start the search from.
    :param keyword: The keyword to search for in folder names.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if keyword in dirname:
                folder_path = os.path.join(dirpath, dirname)
                print(f"Deleting folder: {folder_path}")
                shutil.rmtree(folder_path)