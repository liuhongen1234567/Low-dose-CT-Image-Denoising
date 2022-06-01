import os
from dataset.datasets import CT_Dataset
from utls.loss_ddnet import Loss, GetMask
from utls.eval import test
from models.DD_Net.ddnet_model import DD_Net
from torch.utils import data
import numpy as np
import torch
from torch.optim import lr_scheduler


def main():
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    batch = 8
    threshold = 0.04
    Mask_Enable = False
    device = "cuda"

    root_path = '/Chest'
    train_img_path = '{}/Train/lowdoseCT'.format(root_path)
    train_gt_path = '{}/Train/highdoseCT'.format(root_path)
    test_img_path = '{}/Test/lowdoseCT'.format(root_path)
    test_gt_path = '{}/Test/highdoseCT'.format(root_path)

    train_set = CT_Dataset(train_img_path, train_gt_path, False)
    test_set = CT_Dataset(test_img_path, test_gt_path, False)

    train_loader = data.DataLoader(train_set, batch_size=batch,
                                   shuffle=True, num_workers=0, drop_last=False)

    test_loader = data.DataLoader(test_set, batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=False)
    accuracy_file = open('accuracy.txt','w')
    img, gt = next(iter(train_loader))
    print("train: img {} gt {}".format(img.shape, gt.shape))
    img1, gt1 = next(iter(test_loader))
    print("test: img{} gt {}".format(img1.shape, gt1.shape))

    lr = 1e-4
    model = DD_Net()
    model.to(device)
    save_dir ='pths'
    load_path ='{}/ddnet_save.pth'.format(save_dir)
    save_path = '{}/ddnet_save.pth'.format(save_dir)
    if os.path.exists(load_path):
        print("loading Network")
        model.load_state_dict(torch.load(load_path))
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)
    num_epoch = 160
    scheduler = lr_scheduler.MultiStepLR(opt,milestones=[80, 100,120, 140], gamma=0.5)
    max_PSNR, max_SSIM = 0, 0
    try:
        for epoch in range(num_epoch):
            model.train()
            scheduler.step()
            for i, (img, gt) in enumerate(train_loader):
                img, gt = img.to(device), gt.to(device)
                predict1 = model(img)
                step1_loss, MSE_Loss1, MS_SSIM_Loss1 = Loss(predict1, gt)

                filter_img, filter_gt = GetMask(img,predict1, gt, threshold)

                if Mask_Enable:
                    predict2 = model(filter_img)

                    step2_loss, MSE_Loss2, MS_SSIM_Loss2 = Loss(predict2, filter_gt)
                    total_loss = step1_loss + step2_loss
                else:
                    step2_loss, MSE_Loss2, MS_SSIM_Loss2 = 0, 0, 0
                    total_loss = step1_loss

                opt.zero_grad()
                total_loss.backward()
                opt.step()
                if i % 50 == 0:
                    print(
                        "epoch {}/{} iter: {} Step 1: total loss {:.4f}, mse loss {:.4f}, MS-SSIM {:.4f}, Step 2: total loss {:.4f}, mse loss {:.4f}, MS-SSIM {:.4f}; total_loss {:.4f}, memory {:.4f} ".format(
                            epoch,num_epoch,i, step1_loss, MSE_Loss1,MS_SSIM_Loss1,
                            step2_loss,MSE_Loss2,MS_SSIM_Loss2, total_loss,
                            torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                        ))
                if i % 200 == 0:
                    model.eval()
                    PSNR_tot,SSIM_tot = test(model,test_loader,accuracy_file,epoch,False)
                    if PSNR_tot > max_PSNR:
                        max_PSNR = PSNR_tot
                        max_SSIM = SSIM_tot
                        model_state = model.state_dict()
                        torch.save(model_state, save_path)
                        print("max_PSNR {} max_SSIM {}".format(max_PSNR, max_SSIM))

            model.eval()
            PSNR_tot, SSIM_tot = test(model, test_loader, accuracy_file, epoch,True)
            if PSNR_tot > max_PSNR:
                max_PSNR = PSNR_tot
                max_SSIM = SSIM_tot
                model_state = model.state_dict()
                torch.save(model_state, save_path)
                print("max_PSNR {} max_SSIM {}".format(max_PSNR, max_SSIM))

    except KeyboardInterrupt:
        model_state = model.state_dict()
        torch.save(model_state,save_path)
    print("max_PSNR {} max_SSIM {}".format(max_PSNR, max_SSIM))

if __name__ =="__main__":
    main()