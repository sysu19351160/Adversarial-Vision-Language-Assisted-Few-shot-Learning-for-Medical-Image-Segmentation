import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import cv2
from typing import List
import torch

def prob_mass_function(X: np.ndarray) -> np.ndarray:
    unique, count = np.unique(X, return_counts=True, axis=0)
    return count / len(X)

def entropy(Y) -> float:
    prob = prob_mass_function(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

def entropy_image(I: np.ndarray) -> float:
    return entropy(np.ravel(I))


def probability_dist(I):
    return np.histogramdd(np.ravel(I), bins = 256)[0] / I.size

def kl_divergence(I: np.ndarray, J: np.ndarray) -> float:
    epsilon = 1e-10
    P = probability_dist(I) + epsilon
    Q = probability_dist(J) + epsilon
    return np.where(P != 0, P * np.log2(P / Q), 0).sum()

def js_divergence(I: np.ndarray, J: np.ndarray) -> float:
    epsilon = 1e-10
    P = probability_dist(I) + epsilon
    Q = probability_dist(J) + epsilon
    R=(P+Q)/2
    return 0.5*np.where(P != 0, P * np.log2(P / R), 0).sum() + 0.5*np.where(Q != 0, Q * np.log2(Q / R), 0).sum()


def js_divergence1(I, J):#tensor版本 对已经进行了概率分布化的图像求JS散度
    epsilon = 1e-10
    P = I + epsilon
    Q = J + epsilon
    R=(I+J)/2
    return 0.5*torch.where(P != 0, P * torch.log2(P / R), 0).sum() + 0.5*torch.where(Q != 0, Q * torch.log2(Q / R), 0).sum()