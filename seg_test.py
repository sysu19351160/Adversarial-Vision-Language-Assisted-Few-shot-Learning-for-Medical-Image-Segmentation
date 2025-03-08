import os
from os.path import join
import SimpleITK as sitk
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from collections import OrderedDict
from utils.models.Reg_UNet import UNet_reg
from utils.models.Seg_UNet import UNet_seg
from utils.models.UNetplusplus import UnetPlusPlus
from utils.utils.STN import SpatialTransformer, Re_SpatialTransformer
# from utils.augmentation import SpatialTransform
from utils.utils.dataloader_test_seg_LLM import DatasetFromFolder3D as DatasetFromFolder_train
from utils.utils.losses import gradient_loss, ncc_loss, MSE, dice_loss,DSC
from utils.utils.utils import AverageMeter
from scipy.spatial.distance import directed_hausdorff
from reconstruct import reconstruct
from runllm import Detect
import PIL.Image

class BRBS(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=2,
                 lr=1e-4,
                 epoches=200,
                 iters=150,
                 batch_size=1,
                 is_aug=True,
                 shot=1,
                 test_dir='',
                 labeled_dir='',
                 unlabeled_dir='',
                 checkpoint_dir='',
                 checkpoint_dir1='',
                 result_dir='',
                 model_name='BRBS'):
        super(BRBS, self).__init__()
  
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.is_aug = is_aug
        self.shot = shot
        self.test_dir = test_dir
        self.labeled_dir = labeled_dir
        self.unlabeled_dir = unlabeled_dir

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir1 = checkpoint_dir1
        self.model_name = model_name


        self.stn = SpatialTransformer()  # Spatial Transformer
        self.rstn = Re_SpatialTransformer()  # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)


 
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)
        self.reconstruct = reconstruct()
        self.LLM = Detect(model_path="./saves/llava-1.5-7b-lora")


        if torch.cuda.is_available():
            # self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()
            self.reconstruct = self.reconstruct.cuda()

        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)


        train_dataset = DatasetFromFolder_train(self.test_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_test = DataLoader(train_dataset, batch_size=batch_size)

        self.L_sim = ncc_loss
        self.L_smooth = gradient_loss
        self.L_SeC = DSC
        self.L_I = MSE
        self.L_seg = DSC
        self.L_Mix = MSE

        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_i_log = AverageMeter(name='L_I')
        self.L_SeC_log = AverageMeter(name='L_SeC')

        self.L_seg_log = AverageMeter(name='L_seg')
        self.L_mix_log = AverageMeter(name='L_Mix')


    def test_iterator_seg(self, mi, imgPIL):
        with torch.no_grad():
            img1_n = [str(elem) for elem in imgPIL]
            img1_p = PIL.Image.open(img1_n[0])
            messages1 = self.LLM.generate_messages(img1_p)
            token1 = self.LLM.detect(messages1)
            token1 = token1.unsqueeze(0).unsqueeze(0).cuda()
            token1 = self.reconstruct(token1)
            s_m = self.Seger(mi,token1)
            # s_m = self.Seger(mi)

        return s_m


    def test(self):
        self.Seger.eval()
        self.reconstruct.eval()
        k = 0
        DSC = 0
        avd1 = 0
        for i, (mi, ml, name,imgPIL) in enumerate(self.dataloader_test):
            name = name[0]
            k = k + 1
            if torch.cuda.is_available():
                mi = mi.cuda()
                ml = ml.cuda()
            mi = torch.unsqueeze(mi, 1)
            ml = torch.unsqueeze(ml, 1)

            s_m = self.test_iterator_seg(mi,imgPIL)

            new_label = 1 - ml
            new_lab = torch.cat((ml, new_label), dim=1)
            pre = s_m[:,0,:,:]
            pre = torch.unsqueeze(pre,0)

            loss_seg = self.L_seg(s_m, new_lab)

            dsc = loss_seg.cpu().numpy()
            DSC = DSC + dsc

            pre = torch.squeeze(pre,0)
            pre = torch.squeeze(pre, 0)
            ml = torch.squeeze(ml, 0)
            ml = torch.squeeze(ml, 0)
            Hausdorff_dist3 = directed_hausdorff(ml.cpu().numpy(),pre.cpu().numpy())
            Hausdorff_dist4 = directed_hausdorff(pre.cpu().numpy(),ml.cpu().numpy())
            average_Hausdorff_dist1 = (Hausdorff_dist3[0] + Hausdorff_dist4[0]) / 2
            avd1 = avd1 + average_Hausdorff_dist1

            s_m = s_m.data.cpu().numpy()[0, 0]
            s_m = s_m * 255
            s_m = s_m.astype(np.float32)


            cv2.imwrite((join(self.results_dir, 'predict',  name[:-4] + '.png')), s_m)
            print('DSC:','%.4f'%dsc)
            print('AVD:', '%.4f' % average_Hausdorff_dist1)
        aver_DSC = DSC / k
        AVD = avd1 / k
        print('Average DSC:','%.4f'%aver_DSC)
        print('Average AVD:', '%.4f' % AVD)

    def load(self):

        self.Seger.load_state_dict(
            torch.load(''))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1,2')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    LVA = BRBS()
    LVA.load()
    LVA.test()
