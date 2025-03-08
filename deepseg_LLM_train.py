import os
from os.path import join
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import argparse


from utils.models.Reg_UNet import UNet_reg
from utils.models.Seg_UNet import UNet_seg
from utils.utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.utils.dataloader_LLM_seg import DatasetFromFolder3D as DatasetFromFolder_aug_train
from utils.utils.dataloader_LLM_seg_unlabel import DatasetFromFolder3D as DatasetFromFolder_aug_unlabel
from utils.utils.losses import gradient_loss, ncc_loss, MSE, dice_loss
from utils.utils.utils import AverageMeter
import matplotlib.pyplot as plt
import math
from utils.advbias.augmentor.adv_bias import AdvBias
from utils.advbias.common.utils import _disable_tracking_bn_stats,set_grad
from utils.advbias.augmentor.adv_compose_solver import ComposeAdversarialTransformSolver
import scipy.stats
from utils.utils import FGSM
from PIL import Image
import random
from utils.utils.KL_JS import kl_divergence,js_divergence,js_divergence1
import openpyxl
from reconstruct import reconstruct
from runllm import Detect
import PIL.Image


class BRBS(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=2,
                 lr=1e-4,
                 epoches=200,
                 iters=100,
                 batch_size=20,
                 is_aug=True,
                 shot=1,
                 test_dir='val_set',
                 labeled_dir='',
                 labeled_dir1='',
                 labeled_dir2='',
                 unlabeled_dir='',
                 checkpoint_dir='',
                 checkpoint_dir1 = '',
                 result_dir='',
                 model_name='BRBS_LVA'):
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
        self.labeled_dir1 = labeled_dir1
        self.labeled_dir2 = labeled_dir2
        self.unlabeled_dir = unlabeled_dir
        self.checkvalue = 100
        self.checkvalue1 = 0

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir1 = checkpoint_dir1
        self.model_name = model_name

        self.augmentor_bias = AdvBias(
                 config_dict={'epsilon':0.3, ## magnitude constraints
                 'control_point_spacing':[224//2,224//2],
                 'downscale':2, ## downscale image to save time
                 'data_size':[batch_size,1,224,224],
                 'interpolation_order':3,
                 'init_mode':'random',
                 'space':'log'},debug=False)

        self.stn = SpatialTransformer() # Spatial Transformer
        self.rstn = Re_SpatialTransformer() # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)

        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)
        self.reconstruct = reconstruct()
        self.LLM = Detect(model_path="./saves/llava-1.5-7b-lora")


        if torch.cuda.is_available():
            self.Seger = self.Seger.cuda()
            self.reconstruct = self.reconstruct.cuda()


        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)
        self.optre = torch.optim.Adam(self.reconstruct.parameters(), lr=lr)

        train_dataset = DatasetFromFolder_aug_train(self.labeled_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size)

        unlabeled_dataset = DatasetFromFolder_aug_unlabel(self.labeled_dir2, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_unlabel = DataLoader(unlabeled_dataset, batch_size=batch_size)


        self.L_sim = ncc_loss
        self.L_smooth = gradient_loss
        self.L_SeC = dice_loss
        self.L_I = MSE
        self.L_seg1 = nn.BCEWithLogitsLoss()
        self.KLloss = nn.KLDivLoss()
        self.L_seg = dice_loss
        self.L_Mix = MSE
        self.LLL = [ ]
        self.L_seg_value = 0
        self.L_Sup_value = 0
        self.L_Adv_value = 0
        self.L_Agr_value = 0

        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_i_log = AverageMeter(name='L_I')
        self.L_SeC_log = AverageMeter(name='L_SeC')
        self.L_SUP_log = AverageMeter(name='L_Sup')
        self.L_Adv_log = AverageMeter(name='L_Adv')
        self.L_Agr_log = AverageMeter(name='L_Agr')

        self.L_seg_log = AverageMeter(name='L_seg')
        self.L_mix_log = AverageMeter(name='L_Mix')

    def train_iterator(self, labed_img1, labed_lab1,unlabel_img,img_PIL1,img_PIL3):


        new_img = torch.unsqueeze(labed_img1, 1)
        new_lab = torch.unsqueeze(labed_lab1, 1)
        unlabel_img = torch.unsqueeze(unlabel_img, 1)

        token_list1 = []
        for item in img_PIL1:
            img_p = PIL.Image.open(item)
            messages = self.LLM.generate_messages(img_p)
            token = self.LLM.detect(messages)
            token = token.unsqueeze(0).unsqueeze(0).cuda()
            token = self.reconstruct(token)
            token_list1.append(token)
        token1 = token_list1[0]
        for i in range(len(token_list1) - 1):
            token1 = torch.cat((token1, token_list1[i + 1]), 0)

        token_list3 = []
        for item in img_PIL3:
            img_p = PIL.Image.open(item)
            messages = self.LLM.generate_messages(img_p)
            token = self.LLM.detect(messages)
            token = token.unsqueeze(0).unsqueeze(0).cuda()
            token = self.reconstruct(token)
            token_list3.append(token)
        token3 = token_list3[0]
        for i in range(len(token_list3) - 1):
            token3 = torch.cat((token3, token_list3[i + 1]), 0)


        solver = ComposeAdversarialTransformSolver(
            chain_of_transforms=[self.augmentor_bias],
            divergence_types=['kl', 'contour'],  ### you can also change it to 'mse' for mean squared error loss
            divergence_weights=[1.0, 0.5],
            use_gpu=True,
            debug=True,
        )

        vatdata= unlabel_img.detach().clone()
        vatdata = vatdata.to("cuda:1")
        Dcome_loss,adv_data = solver.adversarial_training(
            data=vatdata, model=self.Seger,
            n_iter=1,
            lazy_load=[True],  ## if set to true, it will use the previous sampled random bias field as initialization.
            optimize_flags=[True], power_iteration=False)

        new_img = new_img.to("cuda:1")
        new_lab = new_lab.to("cuda:1")
        img_fgsm, fgsm_noise = FGSM.get_FGSM(self.Seger, self.L_seg, new_img, new_lab,device_name="cuda:1"


        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to True below in Seger update
        for p in self.reconstruct.parameters():
            p.requires_grad = False


        token1 = token1.to("cuda:1")
        s_gen = self.Seger(new_img,token1)
        s_fgsm = self.Seger(img_fgsm,token1)


        untoken = token3.detach().clone()
        unimg = unlabel_img.detach().clone()
        untoken = untoken.to("cuda:1")
        unimg = unimg.to("cuda:1")
        u_gen = self.Seger(unimg,untoken)
        u_adv = self.Seger(adv_data,untoken)

        lamda_sup = 10
        lamda_div = 1


        new_label = 1 - new_lab
        new_lab = torch.cat((new_lab, new_label), dim=1)


        s_gen = s_gen.to("cuda:0")
        new_lab = new_lab.to("cuda:0")
        s_fgsm = s_fgsm.to("cuda:0")
        s_fgsm1 = s_fgsm.to("cuda:0")
        u_adv = u_adv.to("cuda:0")
        u_gen = u_gen.to("cuda:0")
        Dcome_loss=Dcome_loss.to("cuda:0")

        L_sup = self.L_seg(s_gen, new_lab)

        L_div = self.L_seg(s_fgsm, new_lab) + self.L_seg(u_adv, u_gen) + Dcome_loss

        loss_seg = lamda_sup*L_sup + lamda_div*L_div

        self.checkvalue1 = loss_seg

        self.L_seg_log.update(loss_seg.data, new_img.size(0))
        self.L_SUP_log.update(L_sup.data, new_img.size(0))
        self.L_Adv_log.update(L_div, new_img.size(0))

        self.L_seg_value += loss_seg.data.cpu().numpy()
        self.L_Sup_value += L_sup.data.cpu().numpy()
        self.L_Adv_value += L_div.data.cpu().numpy()
        # self.L_Agr_value += lamda_agr*L_agr.data.cpu().numpy()

        s_m2 = s_gen[0,:,:,:]
        s_m2 = s_m2.data.cpu().numpy()[0]
        s_m2 = s_m2 * 255
        s_m2 = s_m2.astype(np.float32)
        cv2.imwrite((join(self.results_dir, 'predict' + '.png')), s_m2)

        s_m3 = new_lab[0, :, :, :].data.cpu().numpy()[0]
        s_m3 = s_m3 * 255
        s_m3 = s_m3.astype(np.float32)
        cv2.imwrite((join(self.results_dir, 'label' + '.png')), s_m3)

        loss_seg.requires_grad_(True)
        loss_Seg = loss_seg
        loss_Seg.backward()
        self.optS.step()
        self.Seger.zero_grad()
        self.optS.zero_grad()



    def train_epoch(self, epoch):
        self.Seger = nn.DataParallel(self.Seger,device_ids=[1])
        self.Seger.train()



        for i in range(self.iters):
            labed_img, labed_lab, img_PIL = next(self.dataloader_train.__iter__())
            unlabed_img, img_PIL2 = next(self.dataloader_unlabel.__iter__())


            if torch.cuda.is_available():
                labed_img = labed_img.cuda()
                labed_lab = labed_lab.cuda()
                unlabed_img = unlabed_img.cuda()

            self.train_iterator(labed_img, labed_lab,unlabed_img,img_PIL,img_PIL2)
            res1 = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_SUP_log.__str__(),
                             self.L_Adv_log.__str__(),
                             self.L_Agr_log.__str__(),
                             self.L_seg_log.__str__()])
            print(res1)

    def checkpoint(self, epoch, k):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_3_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.reconstruct.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'reconstruct_3_' + self.model_name, epoch + k),
                   _use_new_zipfile_serialization=False)

    def load(self):
        self.Seger.load_state_dict(
            torch.load(''))

        self.reconstruct.load_state_dict(
            torch.load(''))


    def train(self):
        for epoch in range(self.epoches-self.k):
            self.L_smooth_log.reset()
            self.L_sim_log.reset()
            self.L_SeC_log.reset()
            self.L_i_log.reset()
            self.L_SUP_log.reset()
            self.L_Adv_log.reset()
            self.L_Agr_log.reset()
            self.L_seg_log.reset()
            self.L_mix_log.reset()
            self.train_epoch(epoch+self.k)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             self.L_SUP_log.__str__(),
                             self.L_Adv_log.__str__(),
                             self.L_Agr_log.__str__(),
                             self.L_seg_log.__str__()])

            self.L_seg_value = self.L_seg_value / self.iters
            self.L_Sup_value = self.L_Sup_value / self.iters
            self.L_Adv_value = self.L_Adv_value / self.iters
            self.L_Agr_value = self.L_Agr_value / self.iters

            self.LLL.append([self.L_Sup_value,self.L_Adv_value,self.L_Agr_value,self.L_seg_value])
            print(res)

            self.L_seg_value = 0
            self.L_Sup_value = 0
            self.L_Adv_value = 0
            self.L_Agr_value = 0

            if epoch % 5 == 0 and self.checkvalue1 < self.checkvalue:
                self.checkvalue = self.checkvalue1
                self.checkpoint(epoch, self.k)

        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        for row in self.LLL:
            worksheet.append(row)
        workbook.save('train_loss.xlsx')
        workbook.close()

        self.checkpoint(self.epoches-self.k, self.k)


def reposition_batch(path1,path2):
    files = os.listdir(path1)  
    random.seed(1)

    for i in files:
        document = os.path.join(path1, i)  
        document1 = os.path.join(path2, i)

        img = Image.open(document)  
        lab = Image.open(document1)


        x = random.randint(0, 2)
        if x == 0:
            img.save('new_data/labimg/new_image' + os.sep + str(i))  
            lab.save('new_data/labimg/new_label' + os.sep + str(i))
        elif x == 1:
            img.save('new_data/labimg1/new_image' + os.sep + str(i))  
            lab.save('new_data/labimg1/new_label' + os.sep + str(i))
        elif x == 2:
            img.save('new_data/unlabimg/new_image' + os.sep + str(i))
            lab.save('new_data/unlabimg/new_label' + os.sep + str(i))

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1,2')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    LVA = BRBS()
    LVA.load()
    LVA.train()
    LVA.test()