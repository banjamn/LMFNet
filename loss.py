
import torch.nn.functional as F
import torch.nn as nn
import LMFNet.pytorch_ssim as pytorch_ssim
import LMFNet.pytorch_iou as pytorch_iou

bce_loss = nn.BCELoss()
ssim_loss = pytorch_ssim.SSIM()
iou_loss = pytorch_iou.IOU()

def bce_ssim_loss(pred,target):
    pred_S= F.sigmoid(pred)
    bce_out = bce_loss(pred_S,target)
    ssim_out = 1 - ssim_loss(pred_S,target)
    iou_out = iou_loss(pred_S,target)

    loss = bce_out+iou_out+ssim_out
    return loss

def muti_bce_loss_fusion( salList,label):
    loss1 = bce_ssim_loss(salList[0],label)
    loss2 = bce_ssim_loss(salList[1],label)
    loss3 = bce_ssim_loss(salList[2],label)
    loss4 = bce_ssim_loss(salList[3],label)
    loss5 = bce_ssim_loss(salList[4],label)
    loss = loss1+loss2+loss3+loss4+loss5
    return loss


