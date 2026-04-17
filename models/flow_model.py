import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.coupling import AffineCouplingTransform

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Initialize with a random orthogonal matrix to ensure it starts invertible
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        log_det = torch.slogdet(self.weight)[1] * H * W
        weight = self.weight.view(C, C, 1, 1)
        z = F.conv2d(x, weight)
        
        # nflows expects: (output, log_abs_det_jacobian)
        return z, log_det.expand(B)

    def inverse(self, z, context=None):
        B, C, H, W = z.shape
        weight_inv = torch.inverse(self.weight)
        
        # log|det(W^-1)| = -log|det(W)|
        log_det = torch.slogdet(weight_inv)[1] * H * W
        
        weight_inv = weight_inv.view(C, C, 1, 1)
        x = F.conv2d(z, weight_inv)
        
        return x, log_det.expand(B)

class SqueezeTransform(nn.Module):
    """Rearranges [B, C, H, W] to [B, C*4, H/2, W/2] to increase channel depth."""
    def forward(self, x, context=None):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        return x, torch.zeros(B).to(x.device) # log_abs_det_jacobian is 0 for squeeze

    def inverse(self, x, context=None):
        B, C, H, W = x.shape
        C_new = C // 4
        x = x.reshape(B, C_new, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C_new, H * 2, W * 2)
        return x, torch.zeros(B).to(x.device)
    
class GlowCondNet(nn.Module):
    """A Convolutional network for the coupling layers (better for images)."""
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + cond_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        )
        # Initialize last layer to zeros for stability
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, context):
        # Broadcast context [B, 128] to [B, 128, H, W]
        B, C, H, W = x.shape
        context = context.view(B, -1, 1, 1).expand(-1, -1, H, W)
        x_cat = torch.cat([x, context], dim=1)
        return self.net(x_cat)

class GlowModel(nn.Module):
    def __init__(self, img_shape=(3, 96, 96), cond_dim=128, n_levels=3, n_steps=8):
        super().__init__()
        C, H, W = img_shape
        self.img_shape = img_shape
        
        transforms = []
        curr_C, curr_H, curr_W = C, H, W

        for i in range(n_levels):
            transforms.append(SqueezeTransform())
            curr_C *= 4
            curr_H //= 2
            curr_W //= 2

            for _ in range(n_steps):
                transforms.append(ActNorm(curr_C))
                transforms.append(Invertible1x1Conv(curr_C)) # Key Glow component
                transforms.append(AffineCouplingTransform(
                    mask=torch.arange(curr_C) % 2,
                    transform_net_create_fn=lambda in_f, out_f: GlowCondNet(in_f, out_f, cond_dim)
                ))

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([curr_C, curr_H, curr_W])
        )

    def log_prob(self, images, context):
        # images: [B, 3, 96, 96]
        return self.flow.log_prob(images, context=context)

    @torch.no_grad()
    def sample(self, n, context):
        samples = self.flow.sample(n, context=context)
        return samples.clamp(0, 1)

# from nflows.flows import Flow
# from nflows.distributions.normal import StandardNormal
# from nflows.transforms.base import CompositeTransform
# from nflows.transforms.normalization import ActNorm
# from nflows.transforms.coupling import AffineCouplingTransform
# from nflows.transforms.permutations import ReversePermutation
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import numpy as np



# class CondNet(nn.Module):
#     def __init__(self, in_features, out_features, cond_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_features + cond_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, out_features),
#         )

#     def forward(self, x, context):
#         # context: [B, cond_dim], x: [B, in_features]
#         x_cat = torch.cat([x, context], dim=-1)
#         return self.net(x_cat)


# class ConditionalFlow(nn.Module):
#     def __init__(self, img_dim=96*96*3, cond_dim=128, n_layers=4):
#         super().__init__()
#         self.img_dim  = img_dim
#         self.cond_dim = cond_dim

#         # We are now working directly with the pixel dimension
#         flow_dim = img_dim

#         transforms_list = []
#         for _ in range(n_layers):
#             transforms_list += [
#                 ActNorm(flow_dim),
#                 ReversePermutation(features=flow_dim),
#                 AffineCouplingTransform(
#                     mask=torch.arange(flow_dim) % 2, 
#                     # CondNet now receives the split pixel features + context
#                     transform_net_create_fn=lambda in_f, out_f: CondNet(in_f, out_f, cond_dim),
#                 ),
#             ]

#         self.flow = Flow(
#             transform=CompositeTransform(transforms_list),
#             distribution=StandardNormal([flow_dim]),
#         )

#     def log_prob(self, images, context):
#         """images: [B, 3, 96, 96], context: [B, cond_dim]"""
#         # Flatten to [B, 27648]
#         x = images.view(images.size(0), -1) 
#         # Removed self.img_proj(x) line - we feed raw pixels to the flow now
#         return self.flow.log_prob(x, context=context)

#     @torch.no_grad()
#     def sample(self, n, context):
#         h = w = 96 
#         x = self.flow.sample(n, context=context)
#         return x.view(n, 3, h, w).clamp(0, 1)