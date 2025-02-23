import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weight_reduce_loss
from eunet.core import multi_apply


def external_error(pred, mass, external, vert_mask=None, **kwargs):
    x_t1 = pred[:, :3]
    external_loss = -1 * mass * external * x_t1

    external_loss = torch.sum(external_loss, dim=-1, keepdim=True)
    return external_loss

def external_loss(
        pred, mass, external, vert_mask=None,
        weight=None, reduction='mean', avg_factor=None, eps=1e-7, **kwargs):
    loss = external_error(pred, mass, external, vert_mask=vert_mask, **kwargs)

    if weight is not None:
        weight = weight.float()
    if vert_mask is not None:
        avg_factor = torch.sum(vert_mask) + eps if reduction == 'mean' else None
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss,


@LOSSES.register_module()
class ExternalLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_external',):
        super(ExternalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = external_loss
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                mass, external,
                vert_mask=None,
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
            cls_score, mass, external,
            vert_mask if vert_mask is not None else [None]*bs,
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
