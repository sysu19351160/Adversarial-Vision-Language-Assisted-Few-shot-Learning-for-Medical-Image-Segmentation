from os.path import join
from os import listdir
from scipy.io import loadmat
import SimpleITK as sitk
import pandas as pd
from torch.utils import data
import numpy as np
import cv2
# from utils.augmentation_cpu import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir,unlabeled_dir, num_classes, shot=1):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir
        self.unlabeled_dir = unlabeled_dir
        self.num_classes = num_classes
        self.labeled_filenames = self.labeled_filenames[shot:]

    def __getitem__(self, index):
        random_index1 = np.random.randint(low=0, high=len(self.labeled_filenames))  #生成随机数
        labed_img1 = cv2.imread(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index1]),cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        labed_lab1 = cv2.imread(join(self.labeled_file_dir, 'label', self.labeled_filenames[random_index1]),cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        img1_PIL = str(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index1]))

        random_index2 = np.random.randint(low=0, high=len(self.unlabeled_filenames))  # 生成随机数
        labed_img2 = cv2.imread(join(self.unlabeled_dir, 'image', self.unlabeled_filenames[random_index2]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        img2_PIL = str(join(self.unlabeled_dir, 'image', self.unlabeled_filenames[random_index2]))

        return labed_img1, labed_lab1, labed_img2,img1_PIL, img2_PIL


    def __len__(self):
        return len(self.unlabeled_filenames)

