import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from newcrf_layers import NewCRF
import numpy as np
from utils import hconcat_resize
import cv2

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):        
        return self.convblock(x)


def crop_img(source, target): # img menor , img maior (no upsampling)
    # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    diffX = target.size()[2] - source.size()[2]
    diffY = target.size()[3] - source.size()[3]

    # source = F.pad(source, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    # realizando o corte da imagem maior com o tamanho da img menor (proposto no paper do U-net)
    cropped_target = target[:,:,
                diffX//2:target.size()[2] - diffX//2,
                diffY//2:target.size()[3] - diffY//2
            ]
    return cropped_target


# processar 1280 > 1280

# upConv 1280 > 960
# processar 960 + 960 > 960

# upConv 960 > 160
# processar 160 + 160 > 160

# upConv 160 > 112

# conv_block (entrada, praXcanais)
#   processa a entrada convertendo para Xcanais #Conv2D

# decoder_block(entrada, skip_entrada, praXcanais)
#   aumenta a resolucao reduzindo os canais para Xcanais #Conv2DTranspose
#   concatena com a skip
#   vai para conv_block

class Up(nn.Module):
    # upscale e convBlock

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        # dobro os canais porque sera processado a concatenação de 2 imagens
        self.conv = ConvBlock(in_channels*2, out_channels) 

    def forward(self, input, concat_with):

        inter = F.interpolate(input, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        concat = torch.cat([inter, concat_with], dim=1)
        x = self.conv(concat)
        return x

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 16,24,40,80,112,160,960,1280
        # self.bridge = bridge(in_channels=960, out_channels=960) 
        # self.bridge = nn.Conv2d(576, 576, kernel_size=1, stride=1) # bridge
        # self.up0 = Up(in_channels=576, out_channels=96) # 15x20 > 15x20
        # self.up1 = Up(in_channels=96, out_channels=48) # 15x20 > 30x40
        # self.up2 = Up(in_channels=48, out_channels=40) # 30x40 > 30x40
        # self.up3 = Up(in_channels=40, out_channels=24) # 30x40 > 60x80
        # self.up4 = Up(in_channels=24, out_channels=16) # 60x80 > 240x320
        # self.up5 = Up(in_channels=16, out_channels=8) # 240x320 > ??? 480x640?

        # self.conv3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1) # ??? 480x640? > 480x640

        win = 7

        crf_dims = [128, 256, 512, 1024, 1280]  # canais resultantes
        # crf_dims = [32, 64, 128, 256]  # canais resultantes da CRF
        v_dims = [64, 128, 256, 512]      # canais da imagem recebida
        # in_channels = [16, 24, 48, 576] # canais da feature de entrada
        in_channels = [24,32,96,320] # canais da feature de entrada
        # 1280 128
        self.conv0 = nn.Conv2d(crf_dims[4], v_dims[3], kernel_size=1, stride=1) # bridge
        
                        #            feature entra ,            resultado ,              7 , result anterior,  num_heads )
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.conv1 = nn.Conv2d(crf_dims[0], 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, feats):

        # if True: # leitura de tamanho das feats
        #     for block in range(len(feats)):
        #         print("feature[{}]: {}".format(block,feats[block].size()))

        # print("len feats:",len(feats))
  
    # feature[0]: torch.Size([4, 3, 480, 640])
    # feature[1]: torch.Size([4, 32, 240, 320])
    # feature[2]: torch.Size([4, 16, 240, 320])
    # feature[3]: torch.Size([4, 24, 120, 160])
    # feature[4]: torch.Size([4, 24, 120, 160])
    # feature[5]: torch.Size([4, 32, 60, 80])
    # feature[6]: torch.Size([4, 32, 60, 80])
    # feature[7]: torch.Size([4, 32, 60, 80])
    # feature[8]: torch.Size([4, 64, 30, 40])
    # feature[9]: torch.Size([4, 64, 30, 40])
    # feature[10]: torch.Size([4, 64, 30, 40])
    # feature[11]: torch.Size([4, 64, 30, 40])
    # feature[12]: torch.Size([4, 96, 30, 40])
    # feature[13]: torch.Size([4, 96, 30, 40])
    # feature[14]: torch.Size([4, 96, 30, 40])
    # feature[15]: torch.Size([4, 160, 15, 20])
    # feature[16]: torch.Size([4, 160, 15, 20])
    # feature[17]: torch.Size([4, 160, 15, 20])
    # feature[18]: torch.Size([4, 320, 15, 20])
    # feature[19]: torch.Size([4, 1280, 15, 20])


        bridge = self.conv0(feats[19]) # 576 canais
        
        e3 = self.crf3(feats[18], bridge) # [320, 15, 20] | [1280, 15, 20]
        e3p = nn.PixelShuffle(2)(e3) # e3 [1024, 15,20 ]
        e2 = self.crf2(feats[14], e3p) # [96, 30, 40] | [256, 30, 40]
        e2p = nn.PixelShuffle(2)(e2) # e2 [512, 30, 40]
        e1 = self.crf1(feats[7], e2p) # [32, 60, 80] | [128, 60, 80]
        e1p = nn.PixelShuffle(2)(e1) # e1 [256, 60, 80]
        e0 = self.crf0(feats[4], e1p) # [24, 120, 160] |[64, 120, 160]
                                     # e0 [128, 120, 160]
        depth1 = self.sigmoid(self.conv1(e0)) # [1,240,320]
        depth2 = upsample(depth1, scale_factor=2) # [1,240,320] >> tem que ser [1,240,320]


        # import torchvision.transforms as vtransforms
        # imgFeatures = []
        # imgshuffled = []
        # imgCRFed = []

        # imgFeatures.append(np.array(vtransforms.ToPILImage()(feats[3][0,0,:])))
        # imgshuffled.append(np.array(vtransforms.ToPILImage()(bridge[0,0,:])))
        # imgCRFed.append(np.array(vtransforms.ToPILImage()(e3[0,0,:])))

        # imgFeatures.append(np.array(vtransforms.ToPILImage()(feats[2][0,0,:])))
        # imgshuffled.append(np.array(vtransforms.ToPILImage()(e3p[0,0,:])))
        # imgCRFed.append(np.array(vtransforms.ToPILImage()(e2[0,0,:])))

        # imgFeatures.append(np.array(vtransforms.ToPILImage()(feats[1][0,0,:])))
        # imgshuffled.append(np.array(vtransforms.ToPILImage()(e2p[0,0,:])))
        # imgCRFed.append(np.array(vtransforms.ToPILImage()(e1[0,0,:])))

        # imgFeatures.append(np.array(vtransforms.ToPILImage()(feats[0][0,0,:])))
        # imgshuffled.append(np.array(vtransforms.ToPILImage()(e1p[0,0,:])))
        # imgCRFed.append(np.array(vtransforms.ToPILImage()(e0[0,0,:])))

        # # final = []
        # # imgshuffled.append(np.array(vtransforms.ToPILImage()(e0p[0,0,:])))
        # # final.append(np.array(vtransforms.ToPILImage()(depth1[0,:])))
        # imgCRFed.append(np.array(vtransforms.ToPILImage()(depth2[0,:])))

        # cv2.imshow('imgFeatures.jpg', hconcat_resize(imgFeatures))
        # cv2.imshow('imgshuffled.jpg', hconcat_resize(imgshuffled))
        # cv2.imshow('imgCRFed.jpg', hconcat_resize(imgCRFed))
        # # cv2.imshow('final.jpg', hconcat_resize(final))
        
        return depth2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v2( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV2")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        # print(backbone_nn)
        # print("@@@ END BACKBONE @@@")

        #backbone._modules.classifier
        #backbone.classifier._modules
        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.Unet = nn.Sequential(
            Encoder(),
            Decoder()
        )

    def forward(self, x):
        return self.Unet(x)
