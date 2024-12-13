import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from eunet.core import multi_apply


def inertia_error(pred, mass, prev_state, trans, vert_mask=None, dt=1.0, frame_dim=6, **kwargs):
    """
    gt_label is state, prev_state is the input 'state'
    Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Inertia
    x_t1 = pred[:, :3]
    x_t0 = prev_state[:, :3]
    x_tn1 = prev_state[:, frame_dim:frame_dim+3]
    assert prev_state.shape[-1] == frame_dim*2
    v_t0 = (x_t0 - x_tn1)/dt
    if vert_mask is not None:
        v_t0 = v_t0 * vert_mask
    x_arrow = x_t0 + dt*v_t0
    inertia_loss = 0.5 * mass * torch.sum((x_t1-x_arrow)**2, dim=-1, keepdim=True) / (dt**2)

    assert trans.shape[-1] == 9*2
    trans_a = trans[:, frame_dim:frame_dim+3]
    inertia_loss += (mass * x_t1 * trans_a).sum(dim=-1, keepdim=True)

    return inertia_loss

def inertia_loss(
        pred, prev_state, mass, trans, vert_mask=None, dt=1.0, frame_dim=6,
        weight=None, reduction='mean', avg_factor=None, eps=1e-7, **kwargs):
    loss = inertia_error(pred, mass, prev_state, trans, vert_mask, dt, frame_dim, **kwargs)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    if vert_mask is not None:
        avg_factor = torch.sum(vert_mask) + eps if reduction == 'mean' else None
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss,


@LOSSES.register_module()
class InertiaLoss(nn.Module):
    """Cross entropy loss

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_inertia',
                 force_pinned=True,):
        super(InertiaLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.force_pinned = force_pinned

        self.criterion = inertia_loss
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                mass, state, trans, dt, vert_mask=None,
                frame_dim=6,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        bs = len(cls_score)
        loss = multi_apply(
            self.criterion,
            cls_score, state, mass, trans,
            vert_mask if vert_mask is not None and self.force_pinned else [None]*bs,
            dt=dt,
            frame_dim=frame_dim,
            weight=weight, reduction=reduction, avg_factor=avg_factor, **kwargs)[0]
        loss = torch.stack(loss).mean()
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
