import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# class Decoder(nn.Module): #densenet169
#     # The decoder is composed of basic blocks of convolutional 
#     # layers applied on the concatenation of the 2× bilinear upsampling 
#     # of the previous block with the block in the encoder with the same spatial size after upsampling.
#     def __init__(self, num_features=1664, decoder_width = 1.0):
#         super(Decoder, self).__init__()
#         features = int(num_features * decoder_width)
#         print("features:",features)

#         self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

#         self.up1 = UpSample(skip_input=features//1 + 256, output_features=features//2)
#         self.up2 = UpSample(skip_input=features//2 + 128,  output_features=features//4)
#         self.up3 = UpSample(skip_input=features//4 + 64,  output_features=features//8)
#         self.up4 = UpSample(skip_input=features//8 + 64,  output_features=features//16)

#         self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)
#         # self.conv4 = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, features):
#         print("len Features:",len(features))
#         x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
#         x_d0 = self.conv2(F.relu(x_block4))

#         x_d1 = self.up1(x_d0, x_block3)
#         x_d2 = self.up2(x_d1, x_block2)
#         x_d3 = self.up3(x_d2, x_block1)
#         x_d4 = self.up4(x_d3, x_block0)
#         x_d5 = self.conv3(x_d4)
#         return x_d5

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
        # print("xSize: ",x.size())
        # print("concaSize: ",concat_with.size()) 
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        # print("up_x: ",up_x.size())
        x = torch.cat([up_x, concat_with], dim=1)
        return self.UpSample_block(x)
        

class Decoder(nn.Module):
    # https://github.com/alinstein/Depth_estimation/blob/master/Mobile_model.py
    def __init__(self, num_features=960, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)
        # print("features:",features)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1) # 1280 1280
        
        # o "+ algo" significa a soma para permitir o calculo com a concatenacão do residual
        # self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2) # 1600 640
        # self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2) # 800 640
        # self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4) # 704 320
        # self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8) # 352 160
        # self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8) # 184 160
        # self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16) # 176 80

        self.up0 = UpSample(skip_input=features//1 + 160, output_features=features//2) # 1600 640
        self.up1 = UpSample(skip_input=features//2 + 112, output_features=features//2) # 800 640
        self.up2 = UpSample(skip_input=features//2 + 80, output_features=features//4) # 704 320
        self.up3 = UpSample(skip_input=features//4 + 40, output_features=features//8) # 352 160
        self.up4 = UpSample(skip_input=features//8 + 24, output_features=features//8) # 184 160
        self.up5 = UpSample(skip_input=features//8 + 16, output_features=features//16) # 176 80

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1) # 80 1


    def forward(self, features):
        # print("len Features:",len(features)) # são 20 camadas da mobileNet, ele esta escolhendo as features de cada camada especifica
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[7], features[10], features[13],features[16],features[17]

        # for block in range(len(features)):
        #     print("feature[{}]: {}".format(block,features[block].size()))

        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5) # aplicando o residual block
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

# class Decoder(nn.Module):
#     # https://github.com/alinstein/Depth_estimation/blob/master/Mobile_model.py
#     def __init__(self, num_features=1280, decoder_width = 1.0): #0.6 pegou 60% do decoder?
#         super(Decoder, self).__init__()
#         features = int(num_features * decoder_width)
#         # print("features:",features)

#         self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1) # 1280 1280
        
#         # o "+ algo" significa a soma para permitir o calculo com a concatenacão do residual
#         self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2) # 1600 640
#         self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2) # 800 640
#         self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4) # 704 320
#         self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8) # 352 160
#         self.up4 = UpSample(skip_input=features//8 +  24, output_features=features//8) # 184 160
#         self.up5 = UpSample(skip_input=features//8 +  16, output_features=features//16) # 176 80

#         self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1) # 80 1


#     def forward(self, features):
#         print("len Features:",len(features)) # são 20 camadas da mobileNet, ele esta escolhendo as features de cada camada especifica
#         x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]

#         x_d0 = self.conv2(x_block6)
#         x_d1 = self.up0(x_d0, x_block5) # aplicando o residual block
#         x_d2 = self.up1(x_d1, x_block4)
#         x_d3 = self.up2(x_d2, x_block3)
#         x_d4 = self.up3(x_d3, x_block2)
#         x_d5 = self.up4(x_d4, x_block1)
#         x_d6 = self.up5(x_d5, x_block0)
#         x_d7 = self.conv3(x_d6)

#         # [algo,features(canais),width,height]
#         print("x_block6: ",x_block6.shape) # torch.Size([5, 1280, 15, 20])
#         print("x_d0 x_block5: ",x_d0.shape,x_block5.shape) # torch.Size([5, 1280, 17, 22]) torch.Size([5, 320, 15, 20])
#         print("x_d1 x_block4: ",x_d1.shape,x_block4.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 160, 15, 20])
#         print("x_d2 x_block3: ",x_d2.shape,x_block3.shape) # torch.Size([5, 640, 15, 20]) torch.Size([5, 64, 30, 40])
#         print("x_d3 x_block2: ",x_d3.shape,x_block2.shape) # torch.Size([5, 320, 30, 40]) torch.Size([5, 32, 60, 80])
#         print("x_d4 x_block1: ",x_d4.shape,x_block1.shape) # torch.Size([5, 160, 60, 80]) torch.Size([5, 24, 120, 160])
#         print("x_d5 x_block0: ",x_d5.shape,x_block0.shape) # torch.Size([5, 160, 120, 160]) torch.Size([5, 16, 240, 320])
#         print("x_d6: ",x_d6.shape) # torch.Size([5, 80, 240, 320])
#         print("x_d7: ",x_d7.shape) # torch.Size([5, 1, 240, 320])
        
#         return x_d7

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        # backbone_nn = models.mobilenet_v2( pretrained=True )
        backbone_nn = models.mobilenet_v3_large( pretrained=True ) 
        # backbone_nn = models.densenet169( pretrained=True ) # True to transfer learning

        # backbone_nn = models.mobilenet_v2( pretrained=True )

        # modules = list(backbone_nn.children())[:-1]  # delete the last fc layer.
        # backbone_nn = nn.Sequential(*modules)

        
        # print(backbone_nn.classifier)
        # n_inputs = model.fc.in_features
        # # add more layers as required
        # classifier = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(n_inputs, 512))
        # ]))

        # model.fc = classifier



        # FasterRCNN needs to know the number of
        # output channels in a backbone. For resnet101, it's 2048
        print("NOT freezing backbone layers")
        for param in backbone_nn.parameters():
            param.requires_grad = True
        # backbone_nn.out_channels = 1000 # try put on the last layer
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
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )

# class ESPCN(nn.Module):
#     def __init__(self, scale_factor, num_channels=1):
#         super(ESPCN, self).__init__()
#         self.first_part = nn.Sequential(
#             nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
#             nn.Tanh(),
#             nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
#             nn.Tanh(),
#         )
#         self.last_part = nn.Sequential(
#             nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
#             nn.PixelShuffle(scale_factor)
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.in_channels == 32:
#                     nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
#                     nn.init.zeros_(m.bias.data)
#                 else:
#                     nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
#                     nn.init.zeros_(m.bias.data)

#     def forward(self, x):
#         x = self.first_part(x)
#         x = self.last_part(x)
#         return x