"""
.. module:: segat
   :platform: Python
   :synopsis: An Adversarial Attack module for semantic segmentation neural networks in Pytorch.
    针对语义分割神经网络的对抗性攻击模组
              segat (semantic segmentation attacks) requires that the input images are in the form of the network input.
              segat需要输入图像符合网络输入的形式
              For testing a models robustness and adversarial training it advised to avoid the untargeted methods.
              为了测试模型的稳健性和对抗性训练，建议避免使用非目标方法

.. moduleauthor:: Lawrence Stewart <lawrence.stewart@valeo.com>


"""
# from cocoproc import cocodata
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
# from pycocotools.coco import COCO
# from pycocotools import mask
# from cocoproc import custom_transforms
from PIL import Image, ImageFile
from torchvision import transforms as torch_transform
# from cocoproc.utils import decode_segmap, get_pascal_labels
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import pylab
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.patches as mpatches
import torch.nn as nn
# from skimage.measure import compare_ssim as ssim


# TO DOS --- Work out what the default clipping value and alpha value should be
# 计算出默认的裁剪值和阿尔法值应该是什么
# Implement each of the four attacks
# 实施四种攻击


class FGSM():
    """
    FGSM class containing all attacks and miscellaneous functions
    FGSM类包含所有攻击和杂项函数
    """

    def __init__(self, model, loss, alpha=0.03, eps=0.015):
        """ Creates an instance of the FGSM class.
        创建一个FGSM类

        Args:
           model (torch.nn model):  The chosen model to be attacked,
                                    whose output layer returns the nonlinearlity of the pixel
                                    being in each of the possible classes.
           要被攻击的模型
           loss  (function):  The loss function to model was trained on
           被训练的模型的损失函数

        Kwargs:
            alpha (float):  The learning rate of the attack
            攻击的学习率
            eps   (float):  The clipping value for Iterated Attacks
            迭代攻击的裁剪值
        Returns:
           Instance of the FGSM class
           返回FGSM类
        """
        self.model = model
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.predictor = None
        self.default_its = 1
        # set the model to evaluation mode for FGSM attacks
        # 将模型设为评估模式来进行FGSM攻击
        self.model.eval()

    def untargeted(self, img, pred, labels):
        """Performs a single step untargeted FGSM attack
        无目标攻击函数，实施无目标FGSM攻击
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            要被干扰的图片
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            网络对图片的预测
            labels (torch.tensor):  The true labelelling of each pixel in the image
            图片的真实标签
        Returns:
           adv (torch.tensor):  The pertubed image
           返回干扰的图片
           noise (torch.tensor): The adversarial noise added to the image during the attack
           返回添加的噪声
        """

        l = self.loss(pred, labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad = img.grad
        noise = self.alpha * torch.sign(im_grad)
        adv = img + noise
        return adv, noise

    def targeted(self, img, pred, target):
        """Performs a single step targeted FGSM attack
           目标攻击函数，目标FGSM攻击
        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                 for the whole image
            target (torch.tensor):  The target labelling for each pixel
            目标标签


        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        l = self.loss(pred, target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad = img.grad
        noise = -self.alpha * torch.sign(im_grad)
        adv = img + noise
        return adv, noise

    def iterated(self, img, pred, labels, its=None, targeted=False):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI
        FGSMI函数，有目标或无目标的迭代FGSM攻击
        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted
                                    variable)
                                    图像中每个像素的真实标记或我们希望网络将网络错误分类为的目标标记（这应该与目标变量的选择相匹配）

        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted
            选择目标攻击True或无目标攻击False

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        # set number of iterations to be the default value if not given
        # 设置迭代次数，如果没有就是默认值
        its = self.default_its if its is None else its
        adv = img
        tbar = trange(its)
        for i in tbar:
            pred = self.predictor(adv)
            pre = pred[:, 0, :, :]
            pre = torch.unsqueeze(pre, 1)
            l = self.loss(pre, labels)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad = img.grad

            # zero the gradients for the next iteration
            # 将梯度归零，以便于下一次迭代
            self.model.zero_grad()
            # Here the update is GD projected onto ball of radius clipping
            # 这里的更新是GD投影到半径裁剪球上
            if targeted:
                noise = -self.alpha * torch.sign(im_grad).clamp(-self.eps, self.eps)
            else:
                noise = self.alpha * torch.sign(im_grad).clamp(-self.eps, self.eps)

            adv = adv + noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack'.format(i, its))
        return adv, noise

    def iterated_least_likely(self, img, pred, its=None):
        """Performs iterated untargeted FGSM attack towards the
        least likely class, often referred to as FGSMII
        FGSMII攻击函数，针对最不可能的类进行无目标迭代FGSM攻击

        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image

        Kwargs:
            its (int):  The number of iterations to attack
            迭代攻击的次数

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        # set number of iterations to be the default value if not given
        its = self.default_its if its is None else its
        adv = img
        with torch.no_grad():
            pred = self.predictor(adv)
            targets = torch.argmax(pred[0], 0)
            targets = targets.reshape(1, targets.size()[0], -1)
        tbar = trange(its)
        for i in tbar:
            pred = self.predictor(adv)
            l = self.loss(pred, targets)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad = img.grad

            # zero the gradients for the next iteration
            self.model.zero_grad()
            # Here the update is GD projected onto ball of radius clipping
            noise = -self.alpha * torch.sign(im_grad).clamp(-self.eps, self.eps)
            adv = adv + noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMII attack'.format(i, its))
        return adv, noise

    def ssim_iterated(self, img, pred, labels, its=None, targeted=False, threshold=0.99):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI, halfing the current value of
        alpha until the ssim value between the origional and perturbed image
        reaches the threshold value
        ssim攻击函数，执行迭代的无目标或有目标的FGSM攻击，通常称为FGSMI，将alpha的当前值减半，直到原始图像和扰动图像之间的ssim值达到阈值


        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted
                                    variable)

        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted
            threshold (float): Threshold ssi value to obtain
            ssi值的阈值

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        # set number of iterations to be the default value if not given

        ssim_val = 0
        counter = 0
        self.alpha = self.alpha * 2
        its = self.default_its if its is None else its

        tbar = trange(its)

        img_ar = img[0].cpu().detach().numpy()
        img_ar = np.transpose(img_ar, (1, 2, 0))
        img_ar = self.denormalise(img_ar)

        while ssim_val < 0.99:
            self.alpha = self.alpha / 2
            adv = img
            counter += 1
            for i in tbar:
                pred = self.predictor(adv)
                l = self.loss(pred, labels)
                img.retain_grad()
                torch.sum(l).backward()
                im_grad = img.grad

                # zero the gradients for the next iteration
                self.model.zero_grad()
                # Here the update is GD projected onto ball of radius clipping
                if targeted:
                    noise = -self.alpha * torch.sign(im_grad).clamp(-self.eps, self.eps)
                else:
                    noise = self.alpha * torch.sign(im_grad).clamp(-self.eps, self.eps)

                adv = adv + noise
                tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack- attempt {}'.format(i, its, counter))

            # convert to numpy array
            adv_ar = adv[0].cpu().detach().numpy()
            adv_ar = np.transpose(adv_ar, (1, 2, 0))
            adv_ar = self.denormalise(adv_ar)

            ssim_val = ssim(adv_ar, img_ar, multichannel=True)

        return adv, noise

    def untargeted_varied_size(self, img, pred, labels, alphas=[0, 0.005, 0.01]):
        """Performs a single step untargeted FGSM attack for each of the given
        values of alpha.
        根据给出的阿尔法值进行无目标FGSM攻击

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image

        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with
            阿尔法值列表

        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           返回干扰图像列表
           noise_list (torch.tensor list): The list of the adversarial noises created
           返回对抗性噪声列表
        """
        if alphas == []:
            raise Exception("alphas must be a non empty list")
        # create the output lists
        l = self.loss(pred, labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad = img.grad
        noise_list = [alpha * torch.sign(im_grad) for alpha in alphas]
        adv_list = [img + noise for noise in noise_list]
        return adv_list, noise_list

    def targeted_varied_size(self, img, pred, target, alphas=[0, 0.005, 0.01]):
        """Performs a single step targeted FGSM attack for each of the given
        values of alpha.
        根据给出的阿尔法值进行目标FGSM攻击

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel
                                for the whole image
            target (torch.tensor):  The target labelling that we wish the network
                                    to mixclassify the pixel as.

        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with

        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           noise_list (torch.tensor list): The list of the adversarial noises created
        """
        if alphas == []:
            raise Exception("alphas must be a non empty list")
        # create the output lists
        l = self.loss(pred, target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad = img.grad
        noise_list = [-alpha * torch.sign(im_grad) for alpha in alphas]
        adv_list = [img + noise for noise in noise_list]

        return adv_list, noise_list

    def DL3_pred(self, img):
        """Extractor function for deeplabv3 pretained: Please add your own
            to the self.predictor variable to suite your networks output
            已经预训练的deeplabv3提取器函数，请添加你自己的预测函数来预测网络的输出

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            将要被攻击的图片

        Returns: out (torch.tensor): Predicted semantic segmentation
        返回图像的语义分割

        """
        out = self.model(img)
        return out

    def denormalise(self, img):
        """
        Denormalises an image using the image net mean and
        std deviation values.
        去归一化操作

        Args:
            img (numpy array): The image to be denormalised

        Returns:
            img (numpy array): The denormalised image
        """
        img *= (0.229, 0.224, 0.225)
        img += (0.485, 0.456, 0.406)
        img *= 255.000
        img = img.astype(np.uint8).clip(0, 255)
        return img

def get_FGSM(model,loss,img,label,device_name):
    # 选择cuda计算
    device = torch.device(device_name)
    # 将模型加载到cuda上
    model.to(device)
    # 设置模型为评估模式
    model.eval()

    # # 提取样本
    # # 图片x
    # img = img.cpu().numpy()
    # # 标签label
    # gt = label.cpu().numpy()
    #
    # # converts to integer
    # # label转换为int格式
    # tmp = np.array(gt[0]).astype(np.uint8)
    # # decode the segmentation map for plotting
    # # # 解码分割图来进行绘图
    # # # 对转换后的label进行解码
    # # segmap = decode_segmap(tmp, dataset='coco')
    # # 对图片x进行维度转换
    # img_tmp = np.transpose(img[0], axes=[2, 0, 1])  # img_tmp has dim hxwx3
    #
    # # Transpose to the ordering of dimensions expected by Deeplab
    # # 转为deeplab指定的维度排列
    # rimg_tmp = np.transpose(img_tmp, axes=[2, 0, 1])  # has dim 3xhxw
    # rimg_tmp = np.array([rimg_tmp])
    # # 转为tensor格式
    # img_tens = torch.tensor(rimg_tmp, requires_grad=True).type(torch.FloatTensor).to(device)
    # img_tens = torch.tensor(img,requires_grad=True)
    img_tens = img.clone().detach().requires_grad_(True)

    print("predicting segementation using U-Net", end="\n", flush=True)
    # predictedraw=model(torch.tensor(img))
    # 将图片x输入模型，得到预测的输出
    model_out = model(img_tens)
    print("Generating adversarial example", end="\n", flush=True)

    # instantiate the fgsm class
    # 实例化fgsm类，输入模型和损失函数
    attack = FGSM(model, loss)
    # 预测器接口，为输入模型的预测器
    attack.predictor = attack.DL3_pred

    # predict segmentation for the image
    # 预测图像的分割结果
    # the conv output is not normalised between 0 and 1 but is in the form of non-normalised class probabilites
    # conv输出不是在0和1之间归一化的，而是以非归一化类概率的形式
    # 对输入图片x进行预测
    # conv_output = attack.DL3_pred(img_tens)
    # 将int类型的label张量化作为targets
    # targets = torch.tensor([tmp]).type(torch.LongTensor).to(device)

    # 运行对抗性攻击，调用fgsmi，输入图片x和预测y，输出干扰后的图片和噪声
    adv_img_tens, _ = attack.iterated(img_tens, model_out,label)
    # extract to numpy array and convert back to hxwx3
    # 提取到numpy数据并转回hxwx3
    # adv_img_tmp = adv_img_tens[0].cpu().detach().numpy()
    # adv_img_tmp = np.transpose(adv_img_tmp, axes=[1, 2, 0])

    print('FGSM:done')
    return adv_img_tens,_



