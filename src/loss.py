import torch
from math import exp
import torch.nn.functional as F
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

#Structural similarity
# def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
#     L = val_range
#     # luminance, contrast, structure
#     padd = 0
#     # acho q é melhor vc pegar esse loss daquele git q tem milhoes de losses, pra poder todos usarem a mask
#     (_, channel, height, width) = img1.size()
#     if window is None:
#         real_size = min(window_size, height, width)
#         window = create_window(real_size, channel=channel).to(img1.device)

#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2

#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = torch.mean(v1 / v2)  # contrast sensitivity

#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

#     if size_average:
#         ret = ssim_map.mean()
#     else:
#         ret = ssim_map.mean(1).mean(1).mean(1)

#     if full:
#         return ret, cs

#     return ret

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    # from monodepth2
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean()

# class SILogLoss(nn.Module): 
#     def __init__(self,variance_focus=0.85):
#         super(SILogLoss, self).__init__()
#         self.name = 'SILog'
#         self.variance_focus = variance_focus

#     def forward(self, prediction, gt, mask=None, interpolate=True):
#         if interpolate:
#             gt = nn.functional.interpolate(gt, prediction.shape[-2:], mode='bilinear', align_corners=True)

#         if mask is None:
#             mask = (gt > 1e-3).detach()
#             prediction = torch.clamp(prediction, min=1e-6)
        
#         gt = gt[mask]
#         prediction = prediction[mask]
        
#         g = torch.log(gt) - torch.log(prediction)
#         # g = gt - prediction
#         # n, c, h, w = g.shape
#         # norm = 1/(h*w)
#         # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

#         Dg = torch.var(g) + (1-self.variance_focus) * torch.pow(torch.mean(g), 2)
#         return 10 * torch.sqrt(Dg)

class Silog_loss_variance(nn.Module): # https://github.com/SysCV/P3Depth/blob/main/src/losses/loss.py
    def __init__(self, variance_focus=0.85):
        super(Silog_loss_variance, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, prediction, gt):
        # extract the valid region from image
        mask = (gt > 1e-3).detach()
        prediction = torch.clamp(prediction, min=1e-6)
        prediction, gt = prediction[mask], gt[mask]
        d = torch.log(prediction) - torch.log(gt)

        loss = (d ** 2).mean() - self.variance_focus * (d.mean() ** 2)
        return  torch.sqrt(loss) * 10.0 # os papers nao falam sobre sqrt nem *10