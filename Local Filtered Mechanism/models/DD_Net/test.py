import os
from dataset.datasets import CT_Dataset
from utls.eval import test
from models.DD_Net.ddnet_model import DD_Net
from torch.utils import data
import numpy as np
import torch


def main():
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    root_path = '/Chest'
    test_img_path = '{}/Test/lowdoseCT'.format(root_path)
    test_gt_path = '{}/Test/highdoseCT'.format(root_path)

    test_set = CT_Dataset(test_img_path, test_gt_path, False)


    test_loader = data.DataLoader(test_set, batch_size=1,
                                  shuffle=False, num_workers=0, drop_last=False)
    accuracy_file = open('accuracy.txt','w')

    img1, gt1 = next(iter(test_loader))
    print("test: img{} gt {}".format(img1.shape, gt1.shape))

    model = DD_Net()
    model.to(device)
    save_dir ='../../pths'
    save_name ='ddnet_model.pth'
    load_path ='{}/{}'.format(save_dir,save_name)
    if os.path.exists(load_path):
        print("loading Network")
        model.load_state_dict(torch.load(load_path))

    PSNR_tot, SSIM_tot = test(model, test_loader, accuracy_file, 0, False)
    print("PSNR: {} SSIM: {}".format(PSNR_tot,SSIM_tot))




if __name__ =="__main__":
    main()