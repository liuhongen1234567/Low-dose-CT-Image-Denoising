import torch
import torchvision.transforms as transforms
from utls.eval import evaluation
from ddnet_model import DD_Net
import numpy as np
import os
from PIL import Image
import cv2


def save_img(out_path,img):
    denoised_img = img.permute(0, 2, 3, 1)
    denoised_img = denoised_img.to("cpu")
    denoised_img = np.array(denoised_img[0] * 255)
    cv2.imwrite(out_path, denoised_img)
    return None

def main():

    device="cuda"
    G = DD_Net()
    G.to(device)
    save_dir ='../../pths'
    save_name ='model_ddnet.pth'
    load_path ='{}/{}'.format(save_dir,save_name)
    input_dir='../../fig/lowdose'
    high_dose_dir = '../../fig/highdose'
    save_dir ='../../fig/denoised_img'
    transform=transforms.Compose([transforms.ToTensor()])

    if os.path.exists(load_path):
        print("loading Genertor")
        G.load_state_dict(torch.load(load_path))
    G.eval()

    PSNR_t, SSIM_t =0, 0
    for name in os.listdir(input_dir):
        filename=os.path.join(input_dir,name)
        img=Image.open(filename)
        img=transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_img = G(img)

        gt_path = high_dose_dir +'/'+name.replace('L','H')
        gt = Image.open(gt_path)
        gt = transform(gt).unsqueeze(0).to(device)

        PSNR,SSIM = evaluation(denoised_img,gt)
        print("PSNR  {} SSIM {}".format(PSNR,SSIM))
        PSNR_t +=PSNR
        SSIM_t +=SSIM

        save_path = save_dir+'/'+name.replace('L','P')
        save_img(save_path,denoised_img)

    num = len(os.listdir(input_dir))
    print("avarage PSNR {} average SSIM {}".format(PSNR_t/num,SSIM_t/num))

if __name__=='__main__':
    main()