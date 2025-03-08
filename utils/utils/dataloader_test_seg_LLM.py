from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np
import cv2


def is_image_file(filename):
    #返回任意以".png"结尾的文件名
    return any(filename.endswith(extension) for extension in [".png"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, unlabeled_file_dir, num_classes, shot=1):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'Images')) if is_image_file(x)]
        # self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'val_128')) if is_image_file(x)]
        # self.mask_filenames = [x for x in listdir(join(unlabeled_file_dir, 'val_mask_128')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes

    def __getitem__(self, index):
        # random_index = np.random.randint(low=0, high=len(self.labeled_filenames))  #0-100生成随机数
        labed_img = cv2.imread(join(self.labeled_file_dir, 'Images', self.labeled_filenames[index]), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        labed_lab = cv2.imread(join(self.labeled_file_dir, 'Ground-truths', self.labeled_filenames[index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        img_PIL = str(join(self.labeled_file_dir, 'Images', self.labeled_filenames[index]))

        # random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        # val_img = cv2.imread(join(self.unlabeled_file_dir, 'val_128', self.unlabeled_filenames[random_index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        #
        # val_lab = cv2.imread(join(self.unlabeled_file_dir, 'val_mask_128', self.mask_filenames[random_index]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        # labed_img = cv2.imread(join(self.labeled_file_dir, 'val_2018', self.labeled_filenames[index]),cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        #
        # labed_lab = cv2.imread(join(self.labeled_file_dir, 'val_mask_2018', self.labeled_filenames[index]),cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        return labed_img, labed_lab,self.labeled_filenames[index],img_PIL

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.labeled_filenames)

