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

lr = 0.0001


model = PTModel().cuda()
optimizer = torch.optim.Adam( model.parameters(), lr )

checkpoint = torch.load("./checkpoints/epoch-0_loss-0.13168993592262268.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
#model.train()

image = Image.open('data/nyu2_test/00170_colors.png')
image.show()
if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))

# """ 
imsize = 480
loader = vtransforms.Compose([vtransforms.Scale(imsize), vtransforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    print("image_size1:",image.size)
    image = loader(image).float()
    print("image_shape2:",image.shape)
    image = Variable(image, requires_grad=True)
    print("image_shape3:",image.shape)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    print("image_shape4:",image.shape)
    return image.cuda(non_blocking=True)  #assumes that you're using GPU

image = image_loader('data/nyu2_test/00170_colors.png')
print("input_shape:",image.shape)


output = model(image)
output = vtransforms.ToPILImage()(output.int().squeeze(0))
output.show()

# """

# def getNoTransform(is_test=False):
#     return transforms.Compose([
#         ToTensor(is_test=is_test)
#     ])


# transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())
# test_loader = DataLoader(transformed_testing, shuffle=False)

# # sequential = test_loader
# # sample_batched = next(iter(sequential))

# for i, sample_batched in enumerate(test_loader):

#     # Prepare sample and target
#     image = sample_batched['image'].cuda()
#     depth = sample_batched['depth'].cuda(non_blocking=True)

#     # Normalize depth
#     depth_n = DepthNorm( depth )

#     # Predict
#     output = model(image)

#     # LogProgress(model, writer, test_loader, niter)
#     colorize(vutils.make_grid(output.data, nrow=6, normalize=False))
    
