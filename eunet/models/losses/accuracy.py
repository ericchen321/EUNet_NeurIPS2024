# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ACCURACY
from eunet.datasets.utils import GARMENT_TYPE, to_numpy_detach
from eunet.utils import face_normals_batched
from mmcv.ops import QueryAndGroup
from eunet.core import multi_apply
from eunet.utils import MeshViewer
from .compare_loss import cmp_error


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.
    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res

def accuracy_mse(pred, label, indices, indices_type, type_names, prefix='', reduction='sum', default_key='total', overwrite_align=False, vert_mask=None, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses
    mse_error = []
    for i in range(bs):
        m_e = F.mse_loss(pred[i], label[i], reduction='none')
        if vert_mask is not None:
            m_e *= vert_mask[i]
        mse_error.append(m_e)

    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(mse_error, indices, indices_type):
        if reduction == 'sum':
            acc_dict[default_key].append(b_error[0])
            for i in range(1, b_ind.shape[0]):
                assert b_type.shape[-1] == 1, "Only support idx input instead of one hot"
                g_type_idx = b_type[i-1, 0].int()
                g_type = type_names[g_type_idx]
                g_error = b_error[0]
                acc_dict[g_type].append(g_error)
        else:
            for i in range(1, b_ind.shape[0]):
                g_type_idx = torch.argmax(b_type[i-1], dim=0)
                g_type = type_names[g_type_idx]
                start, end = b_ind[i-1], b_ind[i]
                g_error = torch.mean(b_error[start:end])
                acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    if reduction != 'sum' and not overwrite_align:
        # Align the length of keys
        for g_type in type_names:
            key = f"{prefix}.{g_type}"
            if key not in acc_dict.keys():
                acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_l2(pred, label, indices, indices_type, prefix='', **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses
    square_error = [
        torch.sqrt(torch.sum(
            F.mse_loss(pred[i], label[i], reduction='none'), 
            dim=-1)) 
        for i in range(bs)]

    # Calculate per outfit error
    for b_error, b_ind, b_type in zip(square_error, indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            g_error = torch.mean(b_error[start:end])
            acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }
    # Align the length of keys
    for g_type in GARMENT_TYPE:
        key = f"{prefix}.{g_type}"
        if key not in acc_dict.keys():
            acc_dict[key] = torch.zeros(1).cuda()

    return acc_dict

def accuracy_compare_energy(pred, label, indices, indices_type, prefix='', cmp_key=None, hop_mask=None, weight=None, reduction='sum', min_diff=1e-8, eps=1e-7, **kwargs):
    '''
        indices: bs, max(n_outfit) + 1
        from loss/compare_loss.py
    '''
    bs = len(pred)
    acc_dict = defaultdict(list)
    # element-wise losses

    # Calculate per outfit error
    for b_pred, b_label, b_hopmask, b_ind, b_type in zip(pred, label, hop_mask, indices, indices_type):
        for i in range(1, b_ind.shape[0]):
            g_type_idx = torch.argmax(b_type[i-1], dim=0)
            g_type = GARMENT_TYPE[g_type_idx]
            start, end = b_ind[i-1], b_ind[i]
            g_pred = b_pred[start:end]
            g_label = b_label[start:end]
            if b_hopmask is not None:
                g_hopmask = b_hopmask[start:end]
            else:
                g_hopmask = None
            g_error = cmp_error(g_pred, g_label, cmp_key=cmp_key, hop_mask=g_hopmask, weight=weight, reduction=reduction, min_diff=min_diff, eps=eps)[0]
            acc_dict[g_type].append(g_error)
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    return acc_dict


@ACCURACY.register_module()
class MSEAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2',
                 ratio=False):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(MSEAccuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name
        self.ratio = ratio

    def forward(self, pred, target, indices, indices_type=None, type_names=[], overwrite_align=False, vert_mask=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_mse(pred, target, indices, indices_type=indices_type, type_names=type_names, prefix=self.acc_name, reduction=self.reduction, overwrite_align=overwrite_align, vert_mask=vert_mask, ratio=self.ratio, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name

@ACCURACY.register_module()
class L2Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(L2Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, indices, indices_type=None, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_l2(pred, target, indices, indices_type=indices_type, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name
    
@ACCURACY.register_module()
class CompareAccuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_cmp'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(CompareAccuracy, self).__init__()
        self.reduction = reduction

        self._acc_name = acc_name

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_compare_energy(
            pred, target, prefix=self.acc_name, reduction=self.reduction, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name