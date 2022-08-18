import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import torchvision.transforms as vtransforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from PIL import Image
from model import PTModel
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    # print("image_size1:",image.size)
    image = loader(image).float()
    # print("image_shape2:",image.shape)
    image = Variable(image, requires_grad=True)
    # print("image_shape3:",image.shape)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    # print("image_shape4:",image.shape)
    return image.cuda(non_blocking=True)  #assumes that you're using GPU


lr = 0.0001
l1_criterion = nn.L1Loss()

model = PTModel().cuda()
optimizer = torch.optim.Adam( model.parameters(), lr )

#Loading the model
checkpoint = torch.load("./checkpoints/freeze-ep10-loss0.0427.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()


# raw images
image = Image.open('data/nyu2_train/bathroom_0023_out/10.jpg')
depth = Image.open('data/nyu2_train/bathroom_0023_out/10.png')
image.show()
depth.show()
print("image.size:",image.size)
print("depth.size:",depth.size)


# def test(): 
#     test_loss = 0
#     for images, depth in loader:
#         image = image_loader('data/nyu2_train/bathroom_0023_out/10.jpg')
#         depth = image_loader('data/nyu2_train/bathroom_0023_out/10.png')        

#         output = model(image)

#         l_depth = l1_criterion(output, depth_n) 
#         l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
#         loss = (1.0 * l_ssim) + (0.1 * l_depth)
#         test_loss += loss.data.item()
    
#     test_lost /= len(loader)


# """ 
imsize = 960
loader = vtransforms.Compose([vtransforms.Scale(imsize), vtransforms.ToTensor()])


# inicio = time.time()
# execTime = time.time() - inicio


image = image_loader('data/nyu2_train/bathroom_0023_out/10.jpg')
# print("input_shape:",image.shape)

depth = image_loader('data/nyu2_train/bathroom_0023_out/10.png')
depth_n = DepthNorm( depth )


output = model(image)
vtransforms.ToPILImage()(output.int().squeeze(0)).show()




# print("resolução da saida:",output.size)
# diff = torch.abs(output-depth).data

# Compute the loss
l_depth = l1_criterion(output, depth_n) 
l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
loss = (1.0 * l_ssim) + (0.1 * l_depth)
print("loss:",loss.data.item())
