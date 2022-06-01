import os
import cv2
import numpy as np

def adjustWin(img):
    img = img.astype(np.int16)
    low = 100
    high = 300
    img[img == -2000] = 0
    slope=2
    intercept = 0
    img1 = img*slope + intercept
    winWidth = (high - low )
    winCenter = (high + low) / 2
    minWin = float(winCenter) - 0.5 * float(winWidth)
    newImg = (img1 - minWin) / float(winWidth)
    newImg[newImg < 0] = 0
    newImg[newImg > 1] = 1
    newImg = (newImg * 255).astype('uint8')
    return newImg

input_dir = '../fig/denoised_img'
save_dir = './visualization_img'

for name in os.listdir(input_dir):
    filename = os.path.join(input_dir,name)
    img = cv2.imread(filename)
    img1 = adjustWin(img)
    save_name = os.path.join(save_dir,name)
    cv2.imwrite(save_name,img1)


