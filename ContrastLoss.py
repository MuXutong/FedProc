from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F




class SupConLoss_new(nn.Module):

    def __init__(self):
        super(SupConLoss_new, self).__init__()

    def forward(self, features, labels, center):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        center = torch.stack(tuple(center), -1)
        center = F.normalize(center, p=2, dim=0)  # [256,10]

        batch_size = features.shape[0]
        l_sc = None
        for i in range(batch_size):
            current_label = labels[i].item()
            current_feature = features[i]
            current_center = center[:, current_label]
            numerator = torch.exp(current_feature.matmul(current_center))

            denominator = torch.mm(current_feature.view(1, -1), center)
            denominator = torch.sum(torch.exp(denominator))
            if l_sc == None:
                l_sc = -1 * torch.log(numerator / denominator)
            else:
                l_sc += -1 * torch.log(numerator / denominator)

        l_sc = l_sc / batch_size

        return l_sc


