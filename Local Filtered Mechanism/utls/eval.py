import torch.nn.functional as F
from pytorch_ssim import ssim
import math
import torch


def evaluation(predict, gt):
    mse = F.mse_loss(predict, gt).mean()
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    predict = predict.to("cpu")
    gt = gt.to("cpu")
    SSIM = ssim(predict, gt)
    return PSNR, SSIM


def test(model, test_loader, accuracy_file, epoch,isEpoch):
    device = "cuda"
    model.eval()
    PSNR_tot, SSIM_tot, num = 0, 0, 0
    for i, (img_test, gt_test) in enumerate(test_loader):
        img_test = img_test.to(device)
        with torch.no_grad():
            predict = model(img_test)
        predict = predict.to(device)
        gt_test = gt_test.to(device)
        PSNR, SSIM = evaluation(predict, gt_test)
        PSNR_tot += PSNR
        SSIM_tot += SSIM
        num += img_test.shape[0]
    PSNR_tot = PSNR_tot / num
    SSIM_tot = SSIM_tot / num
    print("epoch {} PSNR {},SSIM {} memory {:.8f} ".format(epoch, PSNR_tot, SSIM_tot,
                                                           torch.cuda.max_memory_allocated() / 1024 / 1024,
                                                           ))
    if isEpoch:
        accuracy_file.write('epoch %d PSNR: %f SSIM: %f\n' % (epoch, PSNR_tot, SSIM_tot))
    else:
        accuracy_file.write('epoch_p %d PSNR: %f SSIM: %f\n' % (epoch, PSNR_tot, SSIM_tot))

    return PSNR_tot,SSIM_tot
