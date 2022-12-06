import torch
import torch.nn as nn
import torch.nn.functional as F

# from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
# from .uper_crf_head import PSP

# (version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain)
# (version='large07', inv_depth=False, max_depth=10, pretrained=None)
class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version="tiny07", inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        # self.backbone = SwinTransformer(**backbone_cfg)
        # import torchvision.models as models
        # self.backbone = models.mobilenet_v3_large( pretrained=True ) 
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        in_channels = [96, 192, 384, 768]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        # self.decoder = PSP(**decoder_cfg)
        # self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):

        # feats = [imgs]
        # for k, v in self.backbone.features._modules.items():
        #     feats.append( v(feats[-1]) )

        feats = self.backbone(imgs)
        if self.with_neck:
            feats = self.neck(feats)

        ppm_out = self.decoder(feats)

        #imprime a imagem do crf
        e3 = self.crf3(feats[3], ppm_out)
        e3p = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3p)
        e2p = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2p)
        e1p = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1p)
        # ate aqui

        if self.up_mode == 'mask':
            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth

        # import torchvision.transforms as vtransforms
        # vtransforms.ToPILImage()(feats[3][0,0,:]).show()
        # vtransforms.ToPILImage()(ppm_out[0,0,:]).show()        
        # vtransforms.ToPILImage()(e3[0,0,:]).show()

        # vtransforms.ToPILImage()(feats[2][0,0,:]).show()
        # vtransforms.ToPILImage()(e3p[0,0,:]).show()
        # vtransforms.ToPILImage()(e2[0,0,:]).show()

        # vtransforms.ToPILImage()(feats[1][0,0,:]).show()
        # vtransforms.ToPILImage()(e2p[0,0,:]).show()
        # vtransforms.ToPILImage()(e1[0,0,:]).show()

        # vtransforms.ToPILImage()(feats[0][0,0,:]).show()
        # vtransforms.ToPILImage()(e1p[0,0,:]).show()
        # vtransforms.ToPILImage()(e0[0,0,:]).show()

        # vtransforms.ToPILImage()(depth[0,:]).show()
        return depth