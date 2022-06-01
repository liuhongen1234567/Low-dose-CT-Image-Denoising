import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


def Loss(predict, gt):
    lambda1 = 1
    lambda2 = 0.1
    MSE_Loss = F.mse_loss(predict, gt).mean()
    MS_SSIM_Loss = 1 - ms_ssim(predict, gt, 1)
    total_loss = lambda1 * MSE_Loss + lambda2 * MS_SSIM_Loss
    return total_loss, MSE_Loss, MS_SSIM_Loss


def GetMask(img, predict, gt, thresh):
    diff = torch.abs(predict - gt)
    one = torch.ones_like(diff)
    zero = torch.zeros_like(diff)
    mask = torch.where(diff < thresh, zero, one)
    filtered_img = img * mask
    filtered_gt = gt * mask
    return filtered_img, filtered_gt


if __name__ == '__main__':
    img = torch.rand([3, 1, 512, 512])
    predict = torch.rand([3, 1, 512, 512])
    gt = torch.rand([3, 1, 512, 512])
    filtered_img, filter_gt = GetMask(img, predict, gt, 0.04)
    print(filtered_img.shape, filter_gt.shape)
