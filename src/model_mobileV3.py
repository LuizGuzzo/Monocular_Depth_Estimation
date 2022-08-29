import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class UpSample(nn.Sequential):
    
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()

        self.UpSample_block = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        x = torch.cat([up_x, concat_with], dim=1)
        return self.UpSample_block(x)
        

class Decoder(nn.Module):
    # https://github.com/alinstein/Depth_estimation/blob/master/Mobile_model.py
    def __init__(self, num_features=960, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1) # 1280 1280

        self.up0 = UpSample(skip_input=features//1 + 160, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 112, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 80, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 40, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 + 16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1) # 80 1


    def forward(self, features):
        # print("len Features:",len(features)) # são 20 camadas da mobileNet, ele esta escolhendo as features de cada camada especifica
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[7], features[10], features[13],features[16],features[17]

        # if True: # leitura de tamanho das features
        # for block in range(len(features)):
        #     print("feature[{}]: {}".format(block,features[block].size()))

        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        x_d7 = self.conv3(x_d6)

        # # [???,features(canais),width,height]
        # print("x_block6: ",x_block6.shape) # torch.Size([5, 1280, 15, 20])
        # print("x_d0 x_block5: ",x_d0.shape,x_block5.shape) # torch.Size([5, 1280, 17, 22]) torch.Size([5, 320, 15, 20])
        # print("x_d1 x_block4: ",x_d1.shape,x_block4.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 160, 15, 20])
        # print("x_d2 x_block3: ",x_d2.shape,x_block3.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 64, 30, 40])
        # print("x_d3 x_block2: ",x_d3.shape,x_block2.shape) # torch.Size([5, 320, 30, 40]) torch.Size([5, 32, 60, 80])
        # print("x_d4 x_block1: ",x_d4.shape,x_block1.shape) # torch.Size([5, 160, 60, 80]) torch.Size([5, 24, 120, 160])
        # print("x_d5 x_block0: ",x_d5.shape,x_block0.shape) # torch.Size([5, 160, 120, 160]) torch.Size([5, 16, 240, 320])
        # print("x_d6: ",x_d6.shape) # torch.Size([5, 80, 240, 320])
        # print("x_d7: ",x_d7.shape) # torch.Size([5, 1, 240, 320])
        
        return x_d7


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v3_large( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV3_Large")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        # print(backbone_nn)
        # print("@@@ END BACKBONE @@@")


        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )
