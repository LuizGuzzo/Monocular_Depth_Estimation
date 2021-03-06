import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# class Decoder(nn.Module):
    # The decoder is composed of basic blocks of convolutional 
    # layers applied on the concatenation of the 2× bilinear upsampling 
    # of the previous block with the block in the encoder with the same spatial size after upsampling.
#     def __init__(self, num_features=1664, decoder_width = 1.0):
#         super(Decoder, self).__init__()
#         features = int(num_features * decoder_width)

#         self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

#         self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
#         self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
#         self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
#         self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

#         self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    # def forward(self, features):
    #     x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
    #     x_d0 = self.conv2(F.relu(x_block4))

    #     x_d1 = self.up1(x_d0, x_block3)
    #     x_d2 = self.up2(x_d1, x_block2)
    #     x_d3 = self.up3(x_d2, x_block1)
    #     x_d4 = self.up4(x_d3, x_block0)
    #     return self.conv3(x_d4)

class UpSample(nn.Sequential):
    
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        x = torch.cat([up_x, concat_with], dim=1)
        x = self.convA(x)
        x = self.leakyreluA(x)
        x = self.convB(x)
        x = self.leakyreluB(x)
        return x


class Decoder(nn.Module):
    # https://github.com/alinstein/Depth_estimation/blob/master/Mobile_model.py
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        # poderia por um sequential aqui
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        return self.conv3(x_d6)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        self.original_model = models.mobilenet_v2( pretrained=True )
        # self.original_model = models.densenet169( pretrained=True ) # True to transfer learning

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

