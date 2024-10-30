import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import *

class FeatureLayers(torch.nn.Module):
    def __init__(self, CH=16, use_rgb=False):
        super(FeatureLayers, self).__init__()
        layers = []
        in_channel = 3 if use_rgb else 1
        layers.append(Conv2D(in_channel,CH,5,2,2)) # conv[1]
        layers += [Conv2D(CH,CH,3,1,1) for _ in range(10)] # conv[2-11]
        for d in range(2,5): # conv[12-17]
            layers += [Conv2D(CH,CH,3,1,d,dilation=d) for _ in range(2)]
        layers.append(Conv2D(CH,CH,3,1,1,gn=False,relu=False)) # conv[18]
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, im):
        x = self.layers[0](im)
        for i in range(1,17,2):
            x_ = self.layers[i](x)
            x = self.layers[i+1](x_,residual=x)
        x = self.layers[17](x)
        return x

class SphericalSweep(torch.nn.Module):
    def __init__(self, CH=16, depth_layer = 64):
        super(SphericalSweep, self).__init__()
        self.depth_layer_2 = int(depth_layer / 2)
        self.transfer_conv = Conv2D(CH,CH,3,2,1,gn=False,relu=False)

    def forward(self, feature, grid):
        sweep = [F.grid_sample(feature, grid[:,d,...], align_corners=True) for d in range(self.depth_layer_2)]
        sweep = torch.cat(sweep, 0) # -> (DxN/2) x CH x H x W
        spherical_feature = self.transfer_conv(sweep)
        ND, CH, H_2, W_2 = spherical_feature.shape
        spherical_feature = spherical_feature.view(self.depth_layer_2, -1, CH, H_2, W_2).permute(1,0,2,3,4)
        return spherical_feature

class CostCompute(torch.nn.Module):
    def __init__(self, CH=16, out_CH=8):
        super(CostCompute, self).__init__()
        CH *= 2
        self.fusion = Conv3D(2*CH,CH,3,1,1)
        convs = []
        convs += [Conv3D(CH,CH,3,1,1),
                        Conv3D(CH,CH,3,1,1),
                        Conv3D(CH,CH,3,1,1)]
        convs += [Conv3D(CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,2*CH,3,2,1),
                        Conv3D(2*CH,2*CH,3,1,1),
                        Conv3D(2*CH,2*CH,3,1,1)]
        convs += [Conv3D(2*CH,4*CH,3,2,1),
                        Conv3D(4*CH,4*CH,3,1,1),
                        Conv3D(4*CH,4*CH,3,1,1)]
        self.convs = torch.nn.ModuleList(convs)
        self.deconv1 = DeConv3D(4*CH,2*CH,3,2,1,out_pad=1)
        self.deconv2 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv3 = DeConv3D(2*CH,2*CH,3,2,1,out_pad=1)
        self.deconv4 = DeConv3D(2*CH,CH,3,2,1,out_pad=1)
        self.deconv5 = DeConv3D(CH,out_CH,3,2,1,out_pad=1,gn=False,relu=False)

    def forward(self, feats):
        c = self.fusion(feats)
        c = self.convs[0](c)
        c1 = self.convs[1](c)
        c1 = self.convs[2](c1)
        c = self.convs[3](c)
        c2 = self.convs[4](c)
        c2 = self.convs[5](c2)
        c = self.convs[6](c)
        c3 = self.convs[7](c)
        c3 = self.convs[8](c3)
        c = self.convs[9](c)
        c4 = self.convs[10](c)
        c4 = self.convs[11](c4)
        c = self.convs[12](c)
        c5 = self.convs[13](c)
        c5 = self.convs[14](c5)
        c = self.deconv1(c5, residual=c4)
        c = self.deconv2(c, residual=c3)
        c = self.deconv3(c, residual=c2)
        c = self.deconv4(c, residual=c1)
        costs = self.deconv5(c)
        return costs

class ApprCompute(torch.nn.Module):
    def __init__(self, CH=16, out_CH=8):
        super(ApprCompute, self).__init__()
        self.conv1 = Conv2D(4*CH,2*CH,3,1,1)
        self.conv2 = Conv2D(2*CH,2*CH,3,1,1)
        self.conv3 = Conv2D(2*CH,CH,3,1,1)
        
        self.upscale = DeConv3D(CH,out_CH,3,2,1,out_pad=1)
        
    def forward(self, feats):
        B, C, D, H, W = feats.shape
        feats = feats.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
        a = self.conv1(feats)
        a = self.conv2(a)
        a = self.conv3(a)
        
        a = a.view(B, D, -1, H, W).permute(0, 2, 1, 3, 4)
        appr = self.upscale(a)

        return appr

class OmniVolume(torch.nn.Module):
    def __init__(self, CH = 32, out_CH = 16, depth_layer = 64, use_rgb = True, add_appr = False):
        super(OmniVolume, self).__init__()
        self.opts = Edict()
        self.opts.CH = CH
        self.add_appr = add_appr
        if self.add_appr:
            self.opts.out_CH = int(out_CH / 2)
        else:
            self.opts.out_CH = out_CH
        self.opts.use_rgb = use_rgb
        self.opts.depth_layer = depth_layer
        self.feature_layers = FeatureLayers(self.opts.CH, self.opts.use_rgb)
        self.spherical_sweep = SphericalSweep(self.opts.CH, self.opts.depth_layer)
        self.cost_computes = CostCompute(self.opts.CH, self.opts.out_CH)
        if self.add_appr:
            self.appr_computes = ApprCompute(self.opts.CH, self.opts.out_CH)
        
    def forward(self, images, grids):
        feats = []
        for image_num in range(images.shape[1]):
            image = images[:, image_num, ...]
            feat = self.feature_layers(image)
            feats.append(feat)
        
        sphere_feats = []
        for cam_num, feat in enumerate(feats):
            grid = grids[:, cam_num, ...]
            sphere_feat = self.spherical_sweep(feat, grid)
            sphere_feats.append(sphere_feat)
        
        sphere_feats = torch.cat(sphere_feats, 2).permute(0, 2, 1, 3, 4)
        cost_volume = self.cost_computes(sphere_feats)
        
        if self.add_appr:
            appr_volume = self.appr_computes(sphere_feats)
            volume = torch.cat([cost_volume, appr_volume], dim = 1)
            return volume
        else:
            return cost_volume


class Embedding(nn.Module):
    def __init__(self, N_freqs=10, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF(nn.Module):
    def __init__(self, D=2, W=64, x_freq=10, d_freq=4, 
                       f_channel=16, num_image=4, skips=[1]):
        super(NeRF, self).__init__()        
        x_channel = 3 + 6 * x_freq
        d_channel = 3 + 6 * d_freq
        f_c_channel = f_channel + 3 * num_image
        
        self.skips = skips
        self.x_encoder = Embedding(x_freq)
        self.f_c_encoder = nn.Linear(f_c_channel, W)
        
        self.layers = nn.ModuleList()
        for i in range(D):
            if i == 0:
                layer = nn.Linear(x_channel, W)
            elif i in self.skips:
                layer = nn.Linear(x_channel+W, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU())
            self.layers.append(layer)
        self.sigma_out = nn.Linear(W, 1)
        
        self.d_encoder = Embedding(d_freq)
        self.color_layer = nn.Linear(d_channel+W, W)
        self.color_out = nn.Linear(W, 3)
        
    def forward(self, x, d, f, c):
        x_embed = self.x_encoder(x)
        h_embed = x_embed
        f_c_embed = torch.cat([f, c], dim = -1)
        f_c_embed = self.f_c_encoder(f_c_embed)
        
        for i in range(len(self.layers)):
            if i in self.skips:
                h_embed = torch.cat([x_embed, h_embed], -1)
            h_embed = self.layers[i](h_embed)
            h_embed = h_embed * f_c_embed
        sigma = self.sigma_out(h_embed).squeeze(-1)

        d_embed = self.d_encoder(d)
        h_embed = h_embed * f_c_embed
        d_embed = torch.cat([d_embed, h_embed], dim = -1)
        c_embed = self.color_layer(d_embed)
        color = self.color_out(c_embed)
        color = torch.sigmoid(color)

        return sigma, color
