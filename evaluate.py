import torch
from model import *
from classifier import *
from dataset import *

def kl_categorial(p_prob, q_prob, reduction='mean'):
    if reduction == 'sum':
        return torch.sum(p_prob * torch.log(p_prob/q_prob))
    else:
        return torch.sum(p_prob * torch.log(p_prob/q_prob)) / (1.0 * p_prob.size(0))


def kl_uniform(p_prob, reduction='mean'):
    return kl_categorical(p_prob, torch.ones_like(p_prob) / (1.0 * p_prob.size(1)))


