# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat
import torch.nn.functional as F
import iou3d_cuda
from utils import kitti_utils_torch as kitti_utils
import time
import numpy as np
def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target,dep):
        dep=dep.squeeze(2)
        dep[dep<5]=dep[dep<5]*0.01
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        #losss=torch.abs(pred * mask-target * mask)
        #loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss=torch.abs(pred * mask-target * mask)
        loss=torch.sum(loss,dim=2)*dep
        loss=loss.sum()
        loss = loss / (mask.sum() + 1e-4)

        return loss

# class RegWeightedL1Loss(nn.Module):
#     def __init__(self):
#         super(RegWeightedL1Loss, self).__init__()
#
#     def forward(self, output, mask, ind, target):
#         pred = _transpose_and_gather_feat(output, ind)\
#         mask = mask.float()
#         # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
#         loss = F.l1_loss(pred * mask, target * mask, size_average=False)
#         loss = loss / (mask.sum() + 1e-4)
#         return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class depLoss(nn.Module):
    def __init__(self):
        super(depLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss= torch.log(torch.abs((target * mask)-(pred * mask))).mean()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


class Position_loss(nn.Module):
    def __init__(self, opt):
        super(Position_loss, self).__init__()

        const = torch.Tensor(
            [[-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1],
             [-1, 0], [0, -1], [-1, 0], [0, -1], [-1, 0], [0, -1]])
        self.const = const.unsqueeze(0).unsqueeze(0)  # b,c,2
        self.opt = opt

        self.num_joints = 9

    def forward(self, output, batch,phase=None):
        dim = _transpose_and_gather_feat(output['dim'], batch['ind'])
        rot = _transpose_and_gather_feat(output['rot'], batch['ind'])
        prob = _transpose_and_gather_feat(output['prob'], batch['ind'])
        kps = _transpose_and_gather_feat(output['hps'], batch['ind'])
        rot=rot.detach()### solving............

        b = dim.size(0)
        c = dim.size(1)
        # prior, discard in multi-class training
        # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
        # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
        # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88

        mask = batch['hps_mask']
        mask = mask.float()
        calib = batch['calib']
        opinv = batch['opinv']


        cys = (batch['ind'] / self.opt.output_w).int().float()
        cxs = (batch['ind'] % self.opt.output_w).int().float()
        kps[..., ::2] = kps[..., ::2] + cxs.view(b, c, 1).expand(b, c, self.num_joints)
        kps[..., 1::2] = kps[..., 1::2] + cys.view(b, c, 1).expand(b, c, self.num_joints)

        opinv = opinv.unsqueeze(1)
        opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
        kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
        hom = torch.ones(b, c, 1, 9).cuda()
        kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9)
        kps = torch.bmm(opinv, kps).view(b, c, 2, 9)
        kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18
        si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
        alpha_idx = rot[:, :, 1] > rot[:, :, 5]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
        alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
        alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        alpna_pre = alpna_pre.unsqueeze(2)


        rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
        rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
        rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

        calib = calib.unsqueeze(1)
        calib = calib.expand(b, c, -1, -1).contiguous()
        kpoint = kps
        f = calib[:, :, 0, 0].unsqueeze(2)
        f = f.expand_as(kpoint)
        cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
        cxy = torch.cat((cx, cy), dim=2)
        cxy = cxy.repeat(1, 1, 9)  # b,c,16
        kp_norm = (kpoint - cxy) / f

        l = dim[:, :, 2:3]
        h = dim[:, :, 0:1]
        w = dim[:, :, 1:2]
        cosori = torch.cos(rot_y)
        sinori = torch.sin(rot_y)

        B = torch.zeros_like(kpoint)
        C = torch.zeros_like(kpoint)

        kp = kp_norm.unsqueeze(3)  # b,c,16,1
        const = self.const.cuda()
        const = const.expand(b, c, -1, -1)
        A = torch.cat([const, kp], dim=3)

        B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 1:2] = h * 0.5
        B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 3:4] = h * 0.5
        B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 5:6] = h * 0.5
        B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 7:8] = h * 0.5
        B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 9:10] = -h * 0.5
        B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 11:12] = -h * 0.5
        B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, 13:14] = -h * 0.5
        B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, 15:16] = -h * 0.5
        B[:, :, 16:17] = 0
        B[:, :, 17:18] = 0

        C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, 16:17] = 0
        C[:, :, 17:18] = 0

        B = B - kp_norm * C
        kps_mask = mask
        AT = A.permute(0, 1, 3, 2)
        AT = AT.view(b * c, 3, 18)
        A = A.view(b * c, 18, 3)
        B = B.view(b * c, 18, 1).float()
        # mask = mask.unsqueeze(2)
        pinv = torch.bmm(AT, A)
        pinv = torch.inverse(pinv)  # b*c 3 3
        mask2 = torch.sum(kps_mask, dim=2)
        loss_mask = mask2 > 15
        pinv = torch.bmm(pinv, AT)
        pinv = torch.bmm(pinv, B)
        pinv = pinv.view(b, c, 3, 1).squeeze(3)
        # change the center to kitti center. Note that the pinv is the 3D center point in the camera coordinate system
        pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2

        #min_value_dim = 0.2
        dim_mask = dim<0
        dim = torch.clamp(dim, 0 , 10)
        dim_mask_score_mask = torch.sum(dim_mask, dim=2)
        dim_mask_score_mask = 1 - (dim_mask_score_mask > 0)
        dim_mask_score_mask = dim_mask_score_mask.float()

        box_pred = torch.cat((pinv, dim, rot_y), dim=2).detach()
        loss = (pinv - batch['location'])
        loss_norm = torch.norm(loss, p=2, dim=2)
        loss_mask = loss_mask.float()
        loss = loss_norm * loss_mask
        mask_num = (loss != 0).sum()
        loss = loss.sum() / (mask_num + 1)
        dim_gt = batch['dim'].clone()  # b,c,3
        # dim_gt[:, :, 0] = torch.exp(dim_gt[:, :, 0]) * 1.63
        # dim_gt[:, :, 1] = torch.exp(dim_gt[:, :, 1]) * 1.53
        # dim_gt[:, :, 2] = torch.exp(dim_gt[:, :, 2]) * 3.88
        location_gt = batch['location']
        ori_gt = batch['ori']
        dim_gt[dim_mask] = 0



        gt_box = torch.cat((location_gt, dim_gt, ori_gt), dim=2)
        box_pred = box_pred.view(b * c, -1)
        gt_box = gt_box.view(b * c, -1)

        box_score = boxes_iou3d_gpu(box_pred, gt_box)
        box_score = torch.diag(box_score).view(b, c)
        prob = prob.squeeze(2)
        box_score = box_score * loss_mask * dim_mask_score_mask
        loss_prob = F.binary_cross_entropy_with_logits(prob, box_score.detach(), reduce=False)
        loss_prob = loss_prob * loss_mask * dim_mask_score_mask
        loss_prob = torch.sum(loss_prob, dim=1)
        loss_prob = loss_prob.sum() / (mask_num + 1)
        box_score = box_score * loss_mask
        box_score = box_score.sum() / (mask_num + 1)
        return loss, loss_prob, box_score

class kp_align(nn.Module):
    def __init__(self):
        super(kp_align, self).__init__()

        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])
    def forward(self, output, batch):

        kps = _transpose_and_gather_feat(output['hps'], batch['ind'])
        mask = batch['inv_mask']
        index=self.index_x.cuda()
        x_bottom=torch.index_select(kps,dim=2,index=index[0:4])
        bottom_mask = torch.index_select(mask,dim=2,index=index[0:4]).float()
        x_up=torch.index_select(kps,dim=2,index=index[4:8])
        up_mask = torch.index_select(mask, dim=2, index=index[4:8]).float()
        mask=bottom_mask*up_mask
        loss = F.l1_loss(x_up * mask, x_bottom * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)

        return loss
class kp_conv(nn.Module):
    def __init__(self):
        super(kp_conv, self).__init__()
        self.con1=torch.nn.Conv2d(18,18,3,padding=1)
        # self.con2 = torch.nn.Conv1d(32, 32, 3, padding=1)
        # self.con3 = torch.nn.Conv1d(32, 32, 3, padding=1)
        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])
    def forward(self, output):
        kps = output['hps']
        kps=self.con1(kps)
        return kps

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
