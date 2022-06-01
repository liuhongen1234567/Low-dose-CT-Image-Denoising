import torch
import math
import torch.nn.functional as F
from pytorch_ssim import ssim
import torch.nn as nn

VGG_PATH = 'pths/vgg19-dcbb9e9d.pth'


def Gradient_Loss(predict, gt):
    kernel = torch.tensor([[[
        [-1.0, -1.0, -1.0],
        [0, 0, 0],
        [1.0, 1.0, 1.0],
    ]]], dtype=torch.float32).to("cuda")
    predict_edge = F.conv2d(predict, kernel, stride=1, padding=1)
    gt_edge = F.conv2d(gt, kernel, stride=1, padding=1)
    gradientLoss = F.mse_loss(predict_edge, gt_edge).mean()
    return gradientLoss


def VGG_loss(img, gt, vgg):
    img = torch.cat([img, img, img], dim=1)
    gt = torch.cat([gt, gt, gt], dim=1)
    img = img.to("cuda")
    gt = gt.to("cuda")
    with torch.no_grad():
        img_vgg = vgg(img)
        gt_vgg = vgg(gt)
    w_h_d = img.shape[1] * img.shape[2] * img.shape[3]
    w = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 16]
    MSE = 0
    for i in range(5):
        MSE += w[i] * F.mse_loss(img_vgg[i], gt_vgg[i]).mean()
    perc_loss = MSE
    return perc_loss


def Loss(predict, gt, VGG):
    lambda1 = 1
    lambda2 = 0.1
    MSE_Loss = F.mse_loss(predict, gt).mean()
    Vgg_Loss = VGG_loss(predict, gt, VGG)
    total_loss = lambda1 * MSE_Loss + lambda2 * Vgg_Loss
    return total_loss, MSE_Loss, Vgg_Loss


def GetMask(img, predict, gt, thresh):
    diff = torch.abs(predict - gt)
    one = torch.ones_like(diff)
    zero = torch.zeros_like(diff)
    mask = torch.where(diff < thresh, zero, one)
    filtered_img = img * mask
    filtered_gt = gt * mask
    return filtered_img, filtered_gt


def evaluation(predict, gt):
    mse = F.mse_loss(predict, gt).mean()
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    predict = predict.to("cpu")
    gt = gt.to("cpu")
    SSIM = ssim(predict, gt)
    return PSNR, SSIM


from typing import List, cast


class VGG(nn.Module):

    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 1000,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def make_layers(batch_norm):
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=(1,1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg(pretrained):
    model = VGG(make_layers(batch_norm=False))
    if pretrained:
        vgg_path = VGG_PATH
        model.load_state_dict(torch.load(vgg_path))
    return model


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        self.net = vgg(pretrained=True).features

    def forward(self, x):
        out = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in [3, 8, 15, 22, 29]:
                out.append(x)
        out.append(x)
        return out


if __name__ == '__main__':
    device = "cuda"
    img = torch.rand([1, 1, 64, 64]).to(device)
    predict = torch.rand([1, 1, 64, 64]).to(device)
    gt = torch.rand([1, 1, 64, 64]).to(device)
    vgg = Vgg19()
    vgg.to(device)
    vgg.eval()
    total_loss, MSE_Loss, Vgg_Loss = Loss(predict, gt, vgg)
    print("VGG_LOSS: {}".format(Vgg_Loss))

