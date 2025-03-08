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
from reconstruct import reconstruct

from utils.utils.STN import SpatialTransformer, Re_SpatialTransformer


from utils.utils.dataloader_LLM_reg import DatasetFromFolder3D as DatasetFromFolder3D_train
from utils.utils.losses import gradient_loss, ncc_loss, MSE, dice_loss
from utils.utils.utils import AverageMeter
from runllm import Detect



class BRBS(object):
    def __init__(self, k=5,
                 n_channels=1,
                 n_classes=2,
                 lr=1e-4,
                 epoches=200,
                 iters=100,
                 batch_size=1,
                 is_aug=True,
                 shot=1,
                 test_dir='test_set',
                 labeled_dir='',
                 unlabeled_dir='',
                 checkpoint_dir='',
                 checkpoint_dit1='',
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
        self.unlabeled_dir = unlabeled_dir
        self.checkvalue = 1.21
        self.checkvalue1 = 0

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir1 = checkpoint_dit1
        self.model_name = model_name
        self.stn = SpatialTransformer() # Spatial Transformer
        self.rstn = Re_SpatialTransformer() # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)

        self.Reger = UNet_reg(n_channels=n_channels)
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)
        self.reconstruct = reconstruct()
        self.LLM = Detect(model_path = "./saves/llava-1.5-7b-lora")

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()
            self.reconstruct = self.reconstruct.cuda()

        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)
        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)
        self.optre = torch.optim.Adam(self.reconstruct.parameters(), lr=lr)

  
        train_dataset = DatasetFromFolder3D_train(self.labeled_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size)



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

    def train_iterator(self, labed_img, labed_lab, unlabed_img1, unlabed_img2,img_PIL,img1_PIL,img2_PIL):

        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = True  # they are set to True below in Reger update
        for p in self.reconstruct.parameters():
            p.requires_grad = True

        img1_n = [str(elem) for elem in img_PIL]
        img2_n = [str(elem) for elem in img1_PIL]
        img3_n = [str(elem) for elem in img2_PIL]
        img1_p = PIL.Image.open(img1_n[0])
        img2_p = PIL.Image.open(img2_n[0])
        img3_p = PIL.Image.open(img3_n[0])

        messages1 = self.LLM.generate_messages(img1_p)
        messages2 = self.LLM.generate_messages(img2_p)
        messages3 = self.LLM.generate_messages(img3_p)

        token1 = self.LLM.detect(messages1)
        token1 = token1.unsqueeze(0).unsqueeze(0).cuda()
        token1 = self.reconstruct(token1)
        token2 = self.LLM.detect(messages2)
        token2 = token2.unsqueeze(0).unsqueeze(0).cuda()
        token2 = self.reconstruct(token2)
        token3 = self.LLM.detect(messages3)
        token3 = token3.unsqueeze(0).unsqueeze(0).cuda()
        token3 = self.reconstruct(token3)

        rand = np.random.randint(low=0, high=1)
        if rand == 0:
            img1 = labed_img
            img1_p = token1
            lab1 = labed_lab
            img2 = unlabed_img1
            img2_p = token2
            lab2 = None
        elif rand == 1:
            img1 = labed_img
            img1_p = token1
            lab1 = labed_lab
            img2 = unlabed_img2
            img2_p = token3
            lab2 = None
        else:
            img1 = unlabed_img2
            img1_p = token3
            lab1 = None
            img2 = unlabed_img1
            img2_p = token2
            lab2 = None


        img1 = torch.unsqueeze(img1, 1)
        s_m1 = img1.data.cpu().numpy()[0, 0]
        s_m1 = s_m1 * 255
        s_m1 = s_m1.astype(np.float32)
        cv2.imwrite((join(self.results_dir, 'img1' + '.png')), s_m1)

        img2 = torch.unsqueeze(img2, 1)

        s_m2 = img2.data.cpu().numpy()[0, 0]
        s_m2 = s_m2 * 255
        s_m2 = s_m2.astype(np.float32)
        cv2.imwrite((join(self.results_dir, 'img2' + '.png')), s_m2)

        if lab1 is not None:
            lab1 = torch.unsqueeze(lab1, 1)
            s_m3 = lab1.data.cpu().numpy()[0, 0]
            s_m3 = s_m3 * 255
            s_m3 = s_m3.astype(np.float32)
            cv2.imwrite((join(self.results_dir, 'lab1' + '.png')), s_m3)
        if lab2 is not None:
            lab2 = torch.unsqueeze(lab2, 1)
            s_m4 = lab2.data.cpu().numpy()[0, 0]
            s_m4 = s_m4 * 255
            s_m4 = s_m4.astype(np.float32)
            cv2.imwrite((join(self.results_dir, 'lab2' + '.png')), s_m4)



        w_1_to_2, w_2_to_1, w_label_1_to_2, w_label_2_to_1, flow = self.Reger(img1, img2, lab1, lab2, img1_p, img2_p)

        i_w_2_to_1, i_w_1_to_2, i_w_label_2_to_1, i_w_label_1_to_2, i_flow = self.Reger(img2, img1, lab2, lab1, img2_p, img1_p)



        loss_smooth = self.L_smooth(flow) + self.L_smooth(i_flow)   # smooth loss
        self.L_smooth_log.update(loss_smooth.data, labed_img.size(0))

        loss_sim = self.L_sim(w_1_to_2, img2) + self.L_sim(i_w_2_to_1, img1)    # similarity loss
        self.L_sim_log.update(loss_sim.data, labed_img.size(0))

        token1_2 = w_1_to_2 * 255
        token1_2 = token1_2.type(torch.uint8).squeeze(0).squeeze(0)
        token1_2 = PIL.Image.fromarray(token1_2.data.cpu().numpy())
        messages1_2 = self.LLM.generate_messages(token1_2)
        token1_2 = self.LLM.detect(messages1_2)
        token1_2 = token1_2.unsqueeze(0).unsqueeze(0).cuda()
        token1_2 = self.reconstruct(token1_2)

        token2_1 = i_w_2_to_1 * 255
        token2_1 = token2_1.type(torch.uint8).squeeze(0).squeeze(0)
        token2_1 = PIL.Image.fromarray(token2_1.data.cpu().numpy())
        messages2_1 = self.LLM.generate_messages(token2_1)
        token2_1 = self.LLM.detect(messages2_1)
        token2_1 = token2_1.unsqueeze(0).unsqueeze(0).cuda()
        token2_1 = self.reconstruct(token2_1)

        loss_llm = self.L_llm(token1_2, img2_p)+ self.L_llm(token2_1, img1_p)   
        self.L_llm_log.update(loss_llm.data, labed_img.size(0))

        loss_i = self.L_I(-1*self.stn(flow, flow), i_flow)   # inverse loss
        self.L_i_log.update(loss_i.data, labed_img.size(0))


        if lab1 is not None and lab2 is not None:
            w_1_l = self.stn(lab1, flow)
            w_2_l = self.stn(lab2, i_flow)
            loss_sec = self.L_SeC(w_1_l, lab2) + self.L_SeC(w_2_l, lab1)
        elif lab1 is not None and lab2 is None:
            w_1_l = self.stn(lab1, flow)

            s_2 = self.softmax(self.Seger(img2,img2_p)).detach()
            # s_2 = self.softmax(self.Seger(img2)).detach()

            w_2_s = self.stn(s_2, i_flow)
            loss_sec = self.L_SeC(w_1_l, s_2) + self.L_SeC(w_2_s, lab1)
        elif lab1 is None and lab2 is not None:
            s_1 = self.softmax(self.Seger(img1,img1_p)).detach()

            w_1_s = self.stn(s_1, flow)
            w_2_l = self.stn(lab2, i_flow)
            loss_sec = self.L_SeC(w_1_s, lab2) + self.L_SeC(w_2_l, s_1)
        else:
            s_1 = self.softmax(self.Seger(img1,img1_p)).detach()
            # s_1 = self.softmax(self.Seger(img1)).detach()

            w_1_s = self.stn(s_1, flow)
            s_2 = self.softmax(self.Seger(img2,img2_p)).detach()

            w_2_s = self.stn(s_2, i_flow)
            loss_sec = self.L_SeC(w_1_s, s_2) + self.L_SeC(w_2_s, s_1)

        self.L_SeC_log.update(loss_sec.data, labed_img.size(0))

        loss_Reg = 10*loss_smooth + 100*loss_sim + 1*loss_sec + 10*loss_i + 1*loss_llm
        self.checkvalue1 = loss_sim

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()
        self.optre.step()
        self.reconstruct.zero_grad()
        self.optre.step()

        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to True below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False below in Reger update
        for p in self.reconstruct.parameters():
            p.requires_grad = False

        with torch.no_grad():
            img3 = torch.unsqueeze(labed_img, 1)
            img4 = torch.unsqueeze(unlabed_img1, 1)
            img5 = torch.unsqueeze(unlabed_img2, 1)
            if labed_lab is not None:
                lab3 = torch.unsqueeze(labed_lab, 1)
            w_u_to_l, w_l_to_u, w_label_u_to_l, w_label_l_to_u, flow = self.Reger(img4, img3, None, lab3, token2, token1) # unlabeled image1 --> labeled image
            w_l_to_u2, w_u2_to_l, w_label_l_to_u2, w_label_u2_to_l, flow2 = self.Reger(img3, img5, lab3, None, token1, token3)    # labeled image --> unlabeled image2
            # w_u_to_l, w_l_to_u, w_label_u_to_l, w_label_l_to_u, flow = self.Reger(img4, img3, None, lab3)  # unlabeled image1 --> labeled image
            # w_l_to_u2, w_u2_to_l, w_label_l_to_u2, w_label_u2_to_l, flow2 = self.Reger(img3, img5, lab3, None)  # labeled image --> unlabeled image2
        beta = np.random.beta(0.3, 0.3)
        alpha = np.random.beta(0.3, 0.3)
        sty = beta * (w_u_to_l - img3)
        spa = alpha * flow2
        new_img = self.stn(img3 + sty, spa)
        new_lab = self.stn(lab3, spa)


        w_u_to_u2 = self.stn(w_u_to_l, flow2)
        token4 = w_u_to_u2 * 255
        token4 = token4.type(torch.uint8).squeeze(0).squeeze(0)
        token4 = PIL.Image.fromarray(token4.data.cpu().numpy())
        messages4 = self.LLM.generate_messages(token4)
        token4 = self.LLM.detect(messages4)
        token4 = token4.unsqueeze(0).unsqueeze(0).cuda()
        token4 = self.reconstruct(token4)

        s_w_u_to_u2 = self.Seger(w_u_to_u2,token4)
        s_unlabed_img2 = self.Seger(img5,token3.detach())

        gamma = np.random.beta(0.3, 0.3)
        x_mix = gamma * w_u_to_u2 + (1 - gamma) * img5
        token5 = x_mix * 255
        token5 = token5.type(torch.uint8).squeeze(0).squeeze(0)
        token5 = PIL.Image.fromarray(token5.data.cpu().numpy())
        messages5 = self.LLM.generate_messages(token5)
        token5 = self.LLM.detect(messages5)
        token5 = token5.unsqueeze(0).unsqueeze(0).cuda()
        token5 = self.reconstruct(token5)

        s_mix = self.Seger(x_mix, token5)

        y_mix = gamma * s_w_u_to_u2 + (1 - gamma) * s_unlabed_img2
        loss_mix = self.L_Mix(s_mix, y_mix)
        self.L_mix_log.update(loss_mix.data, img3.size(0))

        token6 = new_img * 255
        token6 = token6.type(torch.uint8).squeeze(0).squeeze(0)
        token6 = PIL.Image.fromarray(token6.data.cpu().numpy())
        messages6 = self.LLM.generate_messages(token6)
        token6 = self.LLM.detect(messages6)
        token6 = token6.unsqueeze(0).unsqueeze(0).cuda()
        token6 = self.reconstruct(token6)

        s_gen = self.Seger(new_img, token6)

        new_lab1 = 1-new_lab
        new_lab = torch.cat((new_lab,new_lab1),dim=1)
        loss_seg = self.L_seg(s_gen, new_lab)
        self.L_seg_log.update(loss_seg.data, img3.size(0))

        loss_Seg = 0.01*loss_mix + loss_seg
        loss_Seg.backward()
        self.optS.step()
        self.Seger.zero_grad()
        self.optS.zero_grad()


    def train_epoch(self, epoch):
        self.Seger.train()
        self.Reger.train()
        self.reconstruct.train()
        for i in range(self.iters):
            labed_img, labed_lab, unlabed_img1, unlabed_img2,img_PIL,img1_PIL,img2_PIL = next(self.dataloader_train.__iter__())


            if torch.cuda.is_available():
                labed_img = labed_img.cuda()
                labed_lab = labed_lab.cuda()
                unlabed_img1 = unlabed_img1.cuda()
                unlabed_img2 = unlabed_img2.cuda()


            self.train_iterator(labed_img, labed_lab, unlabed_img1, unlabed_img2,img_PIL,img1_PIL,img2_PIL)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_sim_log.__str__(),
                             self.L_llm_log.__str__(),
                             self.L_SeC_log.__str__(),
                             self.L_i_log.__str__(),
                             self.L_mix_log.__str__(),
                             self.L_seg_log.__str__()])
            print(res)


    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m

    def test_iterator_reg(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Reg
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def test(self):
        self.Seger.eval()
        self.Reger.eval()
        for i, (mi, ml, name) in enumerate(self.dataloader_test_seg):
            name = name[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
            mi = torch.unsqueeze(mi,1)
            s_m = self.test_iterator_seg(mi)

            s_m = s_m.data.cpu().numpy()[0,0]
            s_m = s_m * 255
            s_m = s_m.astype(np.float32)
            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))


            cv2.imwrite((join(self.results_dir, self.model_name, 'seg', name[:-4]+'.png')),s_m)
            print(name[:-4]+'.png')

        for i, (mi, ml, fi, fl, name1, name2) in enumerate(self.dataloader_test_reg):
            name1 = name1[0]
            name2 = name2[0]
            if name1 is not name2:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()
                    fl = fl.cuda()
                mi = torch.unsqueeze(mi, 1)
                fi = torch.unsqueeze(fi, 1)
                ml = torch.unsqueeze(ml, 1)
                fl = torch.unsqueeze(fl, 1)
                w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(mi, fi, ml, fl)

                flow = flow.data.cpu().numpy()[0]
                w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                # w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)
                w_label_m_to_f = w_label_m_to_f.data.cpu().numpy()[0, 0]

                flow = flow*255
                w_m_to_f = w_m_to_f*255
                w_label_m_to_f = w_label_m_to_f*255

                flow = flow.astype(np.float32)
                w_m_to_f = w_m_to_f.astype(np.float32)
                w_label_m_to_f = w_label_m_to_f.astype(np.float32)

                if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
                    os.makedirs(join(self.results_dir, self.model_name, 'flow'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))

                cv2.imwrite((join(self.results_dir, self.model_name, 'w_m_to_f', name1[:-4]+'_'+name2[:-4]+'.png')), w_m_to_f)

                cv2.imwrite((join(self.results_dir, self.model_name, 'w_label_m_to_f', name1[:-4]+'_'+name2[:-4]+'.png')),w_label_m_to_f)

                print(name1[:-4]+'_'+name2[:-4]+'.png')

    def checkpoint(self, epoch, k):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_2_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_2_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.reconstruct.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'reconstruct_2_' + self.model_name, epoch + k),
                   _use_new_zipfile_serialization=False)

    def load(self):
        self.Seger.load_state_dict(
            torch.load(''))
        self.Reger.load_state_dict(
            torch.load(''))
        self.reconstruct.load_state_dict(
            torch.load(''))


    def train(self):
        for epoch in range(self.epoches-self.k):
            self.L_smooth_log.reset()
            self.L_sim_log.reset()
            self.L_llm_log.reset()
            self.L_SeC_log.reset()
            self.L_i_log.reset()
            self.L_seg_log.reset()
            self.L_mix_log.reset()
            self.train_epoch(epoch+self.k)
            if epoch % 5 == 0 and self.checkvalue1 < self.checkvalue:
                self.checkvalue = self.checkvalue1
                self.checkpoint(epoch, self.k)
        self.checkpoint(self.epoches-self.k, self.k)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    LVA = BRBS()
    LVA.load()
    LVA.train()

