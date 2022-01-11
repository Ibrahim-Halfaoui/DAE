# import torch
# import torch.nn as nn
# from torch.functional import Tensor
# from torch.nn.modules.activation import Tanhshrink
# from timm.models.layers import trunc_normal_
# from functools import partial
#
#
# class Ffn(nn.Module):
#     # feed forward network layer after attention
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x, task_embed=None, level=0):
#         N, L, D = x.shape
#         qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # for decoder's task_embedding of different levels of attention layers
#         if task_embed != None:
#             _N, _H, _L, _D = q.shape
#             task_embed = task_embed.reshape(1, _H, _L, _D)
#             if level == 1:
#                 q += task_embed
#                 k += task_embed
#             if level == 2:
#                 q += task_embed
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(N, L, D)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.norm2 = norm_layer(dim)
#         ffn_hidden_dim = int(dim * ffn_ratio)
#         self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         return x
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn1 = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.norm2 = norm_layer(dim)
#         self.attn2 = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.norm3 = norm_layer(dim)
#         ffn_hidden_dim = int(dim * ffn_ratio)
#         self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x, task_embed):
#         x = x + self.attn1(self.norm1(x), task_embed=task_embed, level=1)
#         x = x + self.attn2(self.norm2(x), task_embed=task_embed, level=2)
#         x = x + self.ffn(self.norm3(x))
#         return x
#
#
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
#                                padding=2, bias=False)
#         self.bn1 = nn.InstanceNorm2d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
#                                padding=2, bias=False)
#         self.bn2 = nn.InstanceNorm2d(channels)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Head(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Head, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.bn1 = nn.InstanceNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.resblock = ResBlock(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.resblock(out)
#
#         return out
#
#
# class PatchEmbed(nn.Module):
#     """ Feature to Patch Embedding
#     input : N C H W
#     output: N num_patch P^2*C
#     """
#
#     def __init__(self, patch_size=1, in_channels=64):
#         super().__init__()
#         self.patch_size = patch_size
#         self.dim = self.patch_size ** 2 * in_channels
#
#     def forward(self, x):
#         N, C, H, W = ori_shape = x.shape
#
#         p = self.patch_size
#         num_patches = (H // p) * (W // p)
#         out = torch.zeros((N, num_patches, self.dim)).to(x.device)
#         i, j = 0, 0
#         for k in range(num_patches):
#             if i + p > W:
#                 i = 0
#                 j += p
#             out[:, k, :] = x[:, :, i:i + p, j:j + p].flatten(1)
#             i += p
#         return out, ori_shape
#
#
# class DePatchEmbed(nn.Module):
#     """ Patch Embedding to Feature
#     input : N num_patch P^2*C
#     output: N C H W
#     """
#
#     def __init__(self, patch_size=1, in_channels=64):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_patches = None
#         self.dim = self.patch_size ** 2 * in_channels
#
#     def forward(self, x, ori_shape):
#         N, num_patches, dim = x.shape
#         _, C, H, W = ori_shape
#         p = self.patch_size
#         out = torch.zeros(ori_shape).to(x.device)
#         i, j = 0, 0
#         for k in range(num_patches):
#             if i + p > W:
#                 i = 0
#                 j += p
#             out[:, :, i:i + p, j:j + p] = x[:, k, :].reshape(N, C, p, p)
#             i += p
#         return out
#
#
# class Tail(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Tail, self).__init__()
#         self.output = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         out = self.output(x)
#         return out
#
#
# class ResNetAE(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#
#     def __init__(self, patch_size=4, in_channels=3, mid_channels=16, num_classes=1000, depth=6,
#                  num_heads=8, ffn_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  norm_layer=nn.LayerNorm):
#         super(ResNetAE, self).__init__()
#
#         self.num_classes = num_classes
#         self.embed_dim = patch_size * patch_size * mid_channels
#         self.head = Head(in_channels, mid_channels)
#         self.patch_embedding = PatchEmbed(patch_size=patch_size, in_channels=mid_channels)
#         self.embed_dim = self.patch_embedding.dim
#         if self.embed_dim % num_heads != 0:
#             raise RuntimeError("Embedding dim must be devided by numbers of heads")
#
#         self.pos_embed = nn.Parameter(torch.zeros(1, (128 // patch_size) ** 2, self.embed_dim))
#         self.task_embed = nn.Parameter(torch.zeros(6, 1, (128 // patch_size) ** 2, self.embed_dim))
#
#         self.encoder = nn.ModuleList([
#             EncoderLayer(
#                 dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
#             for _ in range(depth)])
#         self.decoder = nn.ModuleList([
#             DecoderLayer(
#                 dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
#             for _ in range(depth)])
#
#         self.de_patch_embedding = DePatchEmbed(patch_size=patch_size, in_channels=mid_channels)
#         # tail
#         self.tail = Tail(int(mid_channels), in_channels)
#
#         self.acf = nn.Hardtanh(0, 1)
#
#         trunc_normal_(self.pos_embed, std=.02)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x):
#         x = self.head(x)
#         # x = torch.zeros(1, 16, 128, 128).cuda()
#         x, ori_shape = self.patch_embedding(x)
#         x = x + self.pos_embed[:, :x.shape[1]]
#
#         for blk in self.encoder:
#             x = blk(x)
#
#         for blk in self.decoder:
#             x = blk(x, self.task_embed[0, :, :x.shape[1]])
#
#         x = self.de_patch_embedding(x, ori_shape)
#         x = self.tail(x)
#
#         x = self.acf(x)
#         return x
#####################################################################################################################################
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
##########################################################################################
class ResNetAE(torch.nn.Module):
    def __init__(self):
        super(ResNetAE, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=2, padding=1),  #
            torch.nn.ReLU(True))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),  #
            torch.nn.ReLU(True))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),  #
            torch.nn.ReLU(True))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),  #
            torch.nn.ReLU(True))

        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            torch.nn.Conv2d(256, 256, 2, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True))
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            torch.nn.Conv2d(256, 128, 2, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True))
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            torch.nn.Conv2d(128, 64, 2, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True))
        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            torch.nn.Conv2d(64, 3, 2, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True))
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, 3, stride=1, padding=1),
          )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        coded = self.conv4(x3)

        dx1 = self.deconv1(coded)
        concat1 = torch.cat([dx1, x3], dim=1)
        dx2 = self.deconv2(concat1)
        concat2 = torch.cat([dx2, x2], dim=1)
        dx3 = self.deconv3(concat2)
        concat3 = torch.cat([dx3, x1], dim=1)
        dx4 = self.deconv4(concat3)
        decoded = self.final(dx4)

        return decoded

##########################################################################################
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
#
#         super(ResidualBlock, self).__init__()
#
#         self.residual_block = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                             kernel_size=kernel_size, stride=stride, padding=1),
#             torch.nn.BatchNorm2d(in_channels),
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
#                             kernel_size=kernel_size, stride=stride, padding=1),
#             torch.nn.BatchNorm2d(out_channels),
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#
#     def forward(self, x):
#         return x + self.residual_block(x)
#
#
# class ResNetEncoder(torch.nn.Module):
#     def __init__(self,
#                  n_ResidualBlock=8,
#                  n_levels=4,
#                  input_ch=1,
#                  z_dim=10,
#                  bUseMultiResSkips=True):
#
#         super(ResNetEncoder, self).__init__()
#
#         self.max_filters = 2 ** (n_levels+3)
#         self.n_levels = n_levels
#         self.bUseMultiResSkips = bUseMultiResSkips
#
#         self.conv_list = torch.nn.ModuleList()
#         self.res_blk_list = torch.nn.ModuleList()
#         self.multi_res_skip_list = torch.nn.ModuleList()
#
#         self.input_conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
#                             kernel_size=(3, 3), stride=(1, 1), padding=1),
#             torch.nn.BatchNorm2d(8),
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#
#         for i in range(n_levels):
#             n_filters_1 = 2 ** (i + 3)
#             n_filters_2 = 2 ** (i + 4)
#             ks = 2 ** (n_levels - i)
#
#             self.res_blk_list.append(
#                 torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
#                                       for _ in range(n_ResidualBlock)])
#             )
#
#             self.conv_list.append(
#                 torch.nn.Sequential(
#                     torch.nn.Conv2d(n_filters_1, n_filters_2,
#                                     kernel_size=(2, 2), stride=(2, 2), padding=0),
#                     torch.nn.BatchNorm2d(n_filters_2),
#                     torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 )
#             )
#
#             if bUseMultiResSkips:
#                 self.multi_res_skip_list.append(
#                     torch.nn.Sequential(
#                         torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
#                                         kernel_size=(ks, ks), stride=(ks, ks), padding=0),
#                         torch.nn.BatchNorm2d(self.max_filters),
#                         torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                     )
#                 )
#
#         self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
#                                            kernel_size=(3, 3), stride=(1, 1), padding=1)
#
#     def forward(self, x):
#
#         x = self.input_conv(x)
#
#         skips = []
#         for i in range(self.n_levels):
#             x = self.res_blk_list[i](x)
#             if self.bUseMultiResSkips:
#                 skips.append(self.multi_res_skip_list[i](x))
#             x = self.conv_list[i](x)
#
#         if self.bUseMultiResSkips:
#             x = sum([x] + skips)
#
#         x = self.output_conv(x)
#
#         return x
#
#
# class ResNetDecoder(torch.nn.Module):
#     def __init__(self,
#                  n_ResidualBlock=8,
#                  n_levels=4,
#                  z_dim=10,
#                  output_channels=1,
#                  bUseMultiResSkips=True):
#
#         super(ResNetDecoder, self).__init__()
#
#         self.max_filters = 2 ** (n_levels+3)
#         self.n_levels = n_levels
#         self.bUseMultiResSkips = bUseMultiResSkips
#
#         self.conv_list = torch.nn.ModuleList()
#         self.res_blk_list = torch.nn.ModuleList()
#         self.multi_res_skip_list = torch.nn.ModuleList()
#
#         self.input_conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
#                             kernel_size=(3, 3), stride=(1, 1), padding=1),
#             torch.nn.BatchNorm2d(self.max_filters),
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#
#         for i in range(n_levels):
#             n_filters_0 = 2 ** (self.n_levels - i + 3)
#             n_filters_1 = 2 ** (self.n_levels - i + 2)
#             ks = 2 ** (i + 1)
#
#             self.res_blk_list.append(
#                 torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
#                                       for _ in range(n_ResidualBlock)])
#             )
#
#             self.conv_list.append(
#                 torch.nn.Sequential(
#                     torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
#                                              kernel_size=(2, 2), stride=(2, 2), padding=0),
#                     torch.nn.BatchNorm2d(n_filters_1),
#                     torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                 )
#             )
#
#             if bUseMultiResSkips:
#                 self.multi_res_skip_list.append(
#                     torch.nn.Sequential(
#                         torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
#                                                  kernel_size=(ks, ks), stride=(ks, ks), padding=0),
#                         torch.nn.BatchNorm2d(n_filters_1),
#                         torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                     )
#                 )
#
#         self.output_conv = torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
#                                            kernel_size=(3, 3), stride=(1, 1), padding=1)
#
#     def forward(self, z):
#
#         z = z_top = self.input_conv(z)
#
#         for i in range(self.n_levels):
#             z = self.conv_list[i](z)
#             z = self.res_blk_list[i](z)
#             if self.bUseMultiResSkips:
#                 z += self.multi_res_skip_list[i](z_top)
#
#         z = self.output_conv(z)
#
#         return z
#
#
# class ResNetAE(torch.nn.Module):
#     def __init__(self,
#                  input_shape=(128, 256, 1),
#                  n_ResidualBlock=8,
#                  n_levels=4,
#                  z_dim=128,
#                  bUseMultiResSkips=True):
#         super(ResNetAE, self).__init__()
#
#         self.width = input_shape[0]
#         self.height = input_shape[1]
#         self.c = input_shape[2]
#         self.z_dim = z_dim
#         self.n_levels = n_levels
#
#         self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
#                                      input_ch=self.c, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
#         self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
#                                      output_channels=self.c, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
#
#         # self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
#         # self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)
#
#     def encode(self, x):
#         h = self.encoder(x)
#         return h.view(-1)
#
#     def decode(self, z):
#         z = z.view(-1, self.z_dim, int(self.height/(self.n_levels * 4)), int(self.width/(self.n_levels * 4)))
#         h = self.decoder(z)
#         return h
#
#     def forward(self, x):
#         return self.decode(self.encode(x))

####################################################################################################################################
# class ResNetAE(nn.Module):
#     def __init__(self):
#         super(ResNetAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 3, (2,6), stride=2, padding=1),  # b, 1, 28, 28
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
