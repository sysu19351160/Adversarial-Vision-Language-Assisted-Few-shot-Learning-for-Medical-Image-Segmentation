import os
from os.path import join
import SimpleITK as sitk
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import PIL.Image

from utils.models.Reg_UNet import UNet_reg
from utils.models.Seg_UNet import UNet_seg
from utils.utils.STN import SpatialTransformer, Re_SpatialTransformer

from utils.utils.dataloader_LLM_reg_test import DatasetFromFolder3D as DatasetFromFolder_generate_reg
from utils.utils.losses import gradient_loss, ncc_loss, MSE, dice_loss
from utils.utils.utils import AverageMeter
from reconstruct import reconstruct
from runllm import Detect

class BRBS(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=2,
                 lr=1e-4,
                 epoches=400,
                 iters=100,
                 batch_size=1,
                 is_aug=True,
                 shot=1,
                 test_dir='',
                 labeled_dir='',
                 unlabeled_dir='',
                 checkpoint_dir='',
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
        self.model_name = model_name


        self.stn = SpatialTransformer() # Spatial Transformer
        self.rstn = Re_SpatialTransformer() # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)


        self.Reger = UNet_reg(n_channels=n_channels)
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)
        self.reconstruct = reconstruct()
        self.LLM = Detect(model_path="./saves/llava-1.5-7b-lora")

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()
            self.reconstruct = self.reconstruct.cuda()


        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)
        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)
        self.optre = torch.optim.Adam(self.reconstruct.parameters(), lr=lr)

        generate_dataset_reg = DatasetFromFolder_generate_reg(self.labeled_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.generate_dataset_reg = DataLoader(generate_dataset_reg, batch_size=batch_size)
        self.L_sim = ncc_loss
        self.L_llm = ncc_loss
        self.L_smooth = gradient_loss
        self.L_SeC = dice_loss
        self.L_I = MSE
        self.L_seg = dice_loss
        self.L_seg1 = nn.BCEWithLogitsLoss()
        self.L_Mix = MSE
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_i_log = AverageMeter(name='L_I')
        self.L_SeC_log = AverageMeter(name='L_SeC')
        self.L_llm_log = AverageMeter(name='L_llm')
        self.L_seg_log = AverageMeter(name='L_seg')
        self.L_mix_log = AverageMeter(name='L_Mix')

    def test_iterator_reg(self, mi, fi, ml=None, fl=None, mp=None, fp=None):
        with torch.no_grad():
            # Reg
            img1_PIL = [str(elem) for elem in mp]
            img2_PIL = [str(elem) for elem in fp]

            img1_p = PIL.Image.open(img1_PIL[0])
            img2_p = PIL.Image.open(img2_PIL[0])

            messages1 = self.LLM.generate_messages(img1_p)
            messages2 = self.LLM.generate_messages(img2_p)

            token_m = self.LLM.detect(messages1)
            token_m = token_m.unsqueeze(0).unsqueeze(0).cuda()
            token_m = self.reconstruct(token_m)
            token_f = self.LLM.detect(messages2)
            token_f = token_f.unsqueeze(0).unsqueeze(0).cuda()
            token_f = self.reconstruct(token_f)

            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl, token_m, token_f)
            # w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def test(self):
        self.Reger.eval()
        self.reconstruct.eval()
        for i, (mi, ml, fi, mp, fp) in enumerate(self.generate_dataset_reg):
            fl = None
            num = str(i)
            if mi is not None:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()

                mi = torch.unsqueeze(mi, 1)
                fi = torch.unsqueeze(fi, 1)
                ml = torch.unsqueeze(ml, 1)

                w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(mi, fi, ml, fl, mp, fp)




                flow = flow.data.cpu().numpy()[0]
                w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                w_label_m_to_f = w_label_m_to_f.data.cpu().numpy()[0, 0]


                flow = flow*255
                w_m_to_f = w_m_to_f*255
                w_label_m_to_f = w_label_m_to_f*255

                flow = flow.astype(np.float32)
                w_m_to_f = w_m_to_f.astype(np.float32)
                w_label_m_to_f = w_label_m_to_f.astype(np.float32)


                cv2.imwrite((join(self.results_dir,  'image', 'new_'+num+'.png')), w_m_to_f)

                cv2.imwrite((join(self.results_dir,  'label', 'new_'+num+'.png')),w_label_m_to_f)


    def checkpoint(self, epoch, k):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)

    def load(self):

        self.Reger.load_state_dict(
            torch.load(''))
        self.reconstruct.load_state_dict(
            torch.load(''))



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    LVA = BRBS()
    LVA.load()
    LVA.test()
