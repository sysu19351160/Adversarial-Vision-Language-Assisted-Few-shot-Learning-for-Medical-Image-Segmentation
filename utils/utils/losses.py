import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

# def gradient_loss(s, penalty='l2'):
#     dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
#     dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
#     dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
#
#     if(penalty == 'l2'):
#         dy = dy * dy
#         dx = dx * dx
#         dz = dz * dz
#
#     d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
#     return d / 3.0

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def app_gradient_loss(mask, s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(mask * (dx + dy))
    return d / 2.0
# def app_gradient_loss(mask, s, penalty='l2'):
#     dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
#     dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
#     dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
#
#     if(penalty == 'l2'):
#         dy = dy * dy
#         dx = dx * dx
#         dz = dz * dz
#
#     d = torch.mean(mask * (dx + dy + dz))
#     return d / 3.0

def ncc_loss(I, J, win=None):
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
    
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)
    cc_mean=torch.mean(cc)
    return 1 - cc_mean

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = int(np.prod(win))
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     a = torch.sum(y_true * y_pred, (2, 3, 4))
#     b = torch.sum(y_true**2, (2, 3, 4))
#     c = torch.sum(y_pred**2, (2, 3, 4))
#     dice = (2 * a + smooth) / (b + c + smooth)
#     return torch.mean(dice)

def dice_coef(y_true, y_pred):
    smooth = 1.
    a = torch.sum(y_true * y_pred, (2, 3))
    b = torch.sum(y_true**2, (2, 3))
    c = torch.sum(y_pred**2, (2, 3))
    dice = (2 * a + smooth) / (b + c + smooth)
    return torch.mean(dice)

def dice_loss(y_true, y_pred):
    d = dice_coef(y_true, y_pred)
    return 1 - d

def att_dice(y_true, y_pred):
    dice = dice_coef(y_true, y_pred).detach()
    loss = (1 - dice) ** 2 *(1 - dice)
    return loss

# def masked_dice_loss(y_true, y_pred, mask):
#     smooth = 1.
#     a = torch.sum(y_true * y_pred * mask, (2, 3, 4))
#     b = torch.sum((y_true + y_pred) * mask, (2, 3, 4))
#     dice = (2 * a) / (b + smooth)
#     return 1 - torch.mean(dice)

def masked_dice_loss(y_true, y_pred, mask):
    smooth = 1.
    a = torch.sum(y_true * y_pred * mask, (2, 3))
    b = torch.sum((y_true + y_pred) * mask, (2, 3))
    dice = (2 * a) / (b + smooth)
    return 1 - torch.mean(dice)

def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)

def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth))

def mask_crossentropy(y_pred, y_true, mask):
    smooth = 1e-6
    return -torch.mean(mask * y_true * torch.log(y_pred+smooth))

def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))


def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    loss = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return loss.sum() / N


# v2的另一种代码写法
def diceCoeffv3(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    pred = torch.round(pred)


    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    loss = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return loss.sum() / N


def DSC(pred, gt):
    return diceCoeffv3(pred, gt, activation='none')
