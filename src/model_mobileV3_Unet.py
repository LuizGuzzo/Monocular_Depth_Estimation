import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

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
        # up = self.up(input) 
        # cropped = crop_img(concat_with,up) # invertido, errado
        # deveria estar expandindo o input (Width, hight), mas ja que o mobileNet nao expande a cada reducao de canais,
        # estou adaptando o input pro tamanho do concat
        inter = F.interpolate(input, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        concat = torch.cat([inter, concat_with], dim=1)
        x = self.conv(concat)
        return x

# class bridge(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         # self.bridge = nn.Sequential(
#         #     nn.MaxPool2d(2,2),
#         #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
#         #     nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         # )

#         self.max = nn.MaxPool2d(2,2)
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
#     def forward(self, input):
#         # return self.bridge(input)
#         max = self.max(input)
#         conv = self.conv(max)
#         x = self.trans(conv)
#         return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # 16,24,40,80,112,160,960,1280
        # self.bridge = bridge(in_channels=960, out_channels=960) 
        self.bridge = nn.Conv2d(960, 960, kernel_size=1, stride=1) # bridge
        self.up0 = Up(in_channels=960, out_channels=160) # 15x20 > 15x20
        self.up1 = Up(in_channels=160, out_channels=112) # 15x20 > 30x40
        self.up2 = Up(in_channels=112, out_channels=80) # 30x40 > 30x40
        self.up3 = Up(in_channels=80, out_channels=40) # 30x40 > 60x80
        self.up4 = Up(in_channels=40, out_channels=24) # 60x80 > 120x160
        self.up5 = Up(in_channels=24, out_channels=16) # 120x160 > 240x320
        self.up6 = Up(in_channels=16, out_channels=8) # 240x320 > ???

        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1) # ??? > 480x640

    # def decoder_block(x,skip_input,num_filters):
    #     return nn.Sequential(
    #         # diminui os canais
    #         nn.conv_transpose2d(x, num_filters, kernel_size=3, stride=1, padding=1),
    #         # copia a resolução do skip_input (pega a res do mobilenet)
    #         F.interpolate(x, size=[skip_input.size(2), skip_input.size(3)], mode='bilinear', align_corners=True),
    #         # concatena os canais
    #         torch.cat([x, skip_input], dim=1),
    #         ConvBlock(x,num_filters)
    #     )

    def forward(self, features):
        # print("len Features:",len(features))
        f_block0,       f_block1,    f_block2,     f_block3,     f_block4,    f_block5,    f_block6 = \
        features[2], features[4], features[7], features[11], features[13],features[16],features[17]
        #        16           24           40            80           112          160          960
        # [240,320]    [120,160]      [60,80]       [30,40]       [30,40]      [15,20]      [15,20]

        # 0  1  2  3  4  5  6  7  8  9 10 11  12  13  14  15  16  17
        # 3,16,16,24,24,40,40,40,80,80,80,80,112,112,160,160,160,960
        # if True: # leitura de tamanho das features
        #     for block in range(len(features)):
        #         print("feature[{}]: {}".format(block,features[block].size()))


        x_d0 = self.bridge(f_block6)
        x_d1 = self.up0(x_d0, f_block6) # 960 > 160
        x_d2 = self.up1(x_d1, f_block5) #160 > 112
        x_d3 = self.up2(x_d2, f_block4) #112 > 80
        x_d4 = self.up3(x_d3, f_block3) #80 > 40
        x_d5 = self.up4(x_d4, f_block2) #40 > 24
        x_d6 = self.up5(x_d5, f_block1) #24 > 16
        x_d7 = self.up6(x_d6, f_block0) #16 > 8
        x_d8 = self.conv3(x_d7) # 16 > 1

        # # [batchSize,features(canais),width,height]
        # print("x_block6: ",x_block6.shape) # torch.Size([5, 1280, 15, 20])
        # print("x_d0 x_block5: ",x_d0.shape,x_block5.shape) # torch.Size([5, 1280, 17, 22]) torch.Size([5, 320, 15, 20])
        # print("x_d1 x_block4: ",x_d1.shape,x_block4.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 160, 15, 20])
        # print("x_d2 x_block3: ",x_d2.shape,x_block3.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 64, 30, 40])
        # print("x_d3 x_block2: ",x_d3.shape,x_block2.shape) # torch.Size([5, 320, 30, 40]) torch.Size([5, 32, 60, 80])
        # print("x_d4 x_block1: ",x_d4.shape,x_block1.shape) # torch.Size([5, 160, 60, 80]) torch.Size([5, 24, 120, 160])
        # print("x_d5 x_block0: ",x_d5.shape,x_block0.shape) # torch.Size([5, 160, 120, 160]) torch.Size([5, 16, 240, 320])
        # print("x_d6: ",x_d6.shape) # torch.Size([5, 80, 240, 320])
        # print("x_d7: ",x_d7.shape) # torch.Size([5, 1, 240, 320])
        
        return x_d8


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v3_large( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV3_Large")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        print(backbone_nn)
        print("@@@ END BACKBONE @@@")


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
