from os.path import join
from os import listdir

from torch.utils import data
import numpy as np
import cv2
from skimage import util


def is_image_file(filename):
    #返回任意以".png"结尾的文件名
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, unlabeled_file_dir, num_classes, shot=1):
        super(DatasetFromFolder3D, self).__init__()
        # self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'new_img_224')) if is_image_file(x)]
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        # self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'val_128')) if is_image_file(x)]
        # self.mask_filenames = [x for x in listdir(join(unlabeled_file_dir, 'val_mask_128')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.labeled_filenames))  #0-1900生成随机数
        labed_img = cv2.imread(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index]), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        labed_lab = cv2.imread(join(self.labeled_file_dir, 'label', self.labeled_filenames[random_index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        img_PIL = str(join(self.labeled_file_dir, 'image', self.labeled_filenames[random_index]))

        random_aug = np.random.randint(low=0, high=9)
        # random_aug = 2
        if random_aug == 0:
            #只进行旋转缩放
            labed_img,labed_lab = img_rotate_scale(labed_img,labed_lab)
        elif random_aug == 1:
            labed_img, labed_lab = img_rotate_scale(labed_img, labed_lab)
        elif random_aug == 2:
            labed_img, labed_lab = img_rotate_scale(labed_img, labed_lab)
        elif random_aug == 3:
            labed_img, labed_lab = img_rotate_scale(labed_img, labed_lab)
            labed_img = add_Gaussnoise(labed_img)
        elif random_aug == 4:
            labed_img, labed_lab = img_rotate_scale(labed_img, labed_lab)
            labed_img = bilateralfilter(labed_img)


        # random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        # val_img = cv2.imread(join(self.unlabeled_file_dir, 'val_128', self.unlabeled_filenames[random_index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        #
        # val_lab = cv2.imread(join(self.unlabeled_file_dir, 'val_mask_128', self.mask_filenames[random_index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        # cv2.imwrite('C:/Users/Chenye/Desktop/img_aug.png',labed_img*255)
        # cv2.imwrite('C:/Users/Chenye/Desktop/lab_aug.png', labed_lab*255)
        return labed_img.astype(np.float32), labed_lab.astype(np.float32), img_PIL

    def __len__(self):
        return len(self.labeled_filenames)

def img_rotate_scale(img,lab):
#随机旋转缩放图像
    rows, cols =img.shape[:2]
    random_scale = np.random.uniform(0.85,1.15)#随机缩放系数
    random_rotate = np.random.uniform(-40,40)#随机旋转角度
    M = cv2.getRotationMatrix2D((cols/2,rows/2),random_rotate,random_scale)
    # 自适应图片边框大小
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = rows * sin + cols * cos
    new_h = rows * cos + cols * sin
    M[0, 2] += (new_w - cols) * 0.5
    M[1, 2] += (new_h - rows) * 0.5
    w = int(np.round(new_w))
    h = int(np.round(new_h))
    img_aug = cv2.warpAffine(img, M, (w, h))
    lab_aug = cv2.warpAffine(lab, M, (w, h))
    img_aug = cv2.resize(img_aug,(cols,rows))
    lab_aug = cv2.resize(lab_aug,(cols,rows))
    return img_aug, lab_aug

def add_Gaussnoise(img):
    #添加高斯噪声
    img_aug = util.random_noise(img,mode='gaussian',mean = 0,var= 0.00001)
    return img_aug

def bilateralfilter(img):
    #双边滤波
    img_aug = cv2.bilateralFilter(src=img, d=5, sigmaColor= 10,sigmaSpace=10)
    return img_aug