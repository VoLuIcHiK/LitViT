import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from tqdm.notebook import tqdm
import os
import lightning as L
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # нарезаем исходное изображение на кусочки размером 16 на 16 с помощью свертки
        self.patch_embeddings = nn.Conv2d(in_channels=in_chans,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        #создать эмбеддинг для класса
        self.cls_token = nn.Parameter(torch.randn([1, 1, embed_dim]))
        #создать позиционный эмбеддинг
        self.pos_embeddings = nn.Parameter(torch.rand([1, (img_size // patch_size) ** 2 + 1, embed_dim]))

    def forward(self, image):
        # Создание patch embeddinga
        #для flatten указываем позицию (индекс числа), после которой должно сливаться
        #(1, 768, 14, 14) -> (1, 768, 196)
        patches = self.patch_embeddings(image).flatten(start_dim=2).transpose(1, 2)
        b, n, _ = patches.size()
        #добавить эмбеддинг класса
        cls_tokens = self.cls_token.expand(b, -1, -1)
        patch_embeddings = torch.cat((cls_tokens, patches), dim=1)
        #Объединение с positional embeddings
        combined_embeddings = patch_embeddings + self.pos_embeddings
        return combined_embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.2):
        super().__init__()
        # Linear Layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Linear(in_features=hidden_features,
                      out_features=out_features),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, img_size=224, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(in_features=dim,
                             out_features=dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):

        # Attention
        x = self.qkv(x)
        x = reduce(x, 'b p (c h s) ->b p c h s', 'mean', c=3, h=12)
        v, q, k = torch.split(x, 1, dim=2)
        v = rearrange(v, 'b p c h s-> b c h p s')
        q = rearrange(q, 'b p c h s-> b c h p s')
        k = rearrange(k, 'b p c h s-> b c h p s')
        #print(v.size(), q.size(), k.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Out projection
        out = attn @ v
        out = rearrange(out, 'b c h p s -> b p (c h s)')
        x = self.out(out)
        x = self.out_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        # Attention
        self.attn = Attention(dim, num_heads)
        # Dropout
        self.drop = nn.Dropout(drop_rate)
        # Normalization
        self.norm2 = nn.LayerNorm(dim)
        # MLP
        self.mlp = MLP(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       out_features=dim)

    def forward(self, x):
        skip_1 = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + skip_1
        skip_2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + skip_2
        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

from torch.nn.modules.normalization import LayerNorm

class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0.):
        super().__init__()
        # Присвоение переменных
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = patch_size ** 2 * in_chans
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embeddings = PatchEmbedding(img_size=self.img_size,
                                               patch_size=self.patch_size,
                                               in_chans=self.in_chans,
                                               embed_dim=self.embed_dim)
        # Transformer Encoder
        self.transformer_encoder = Transformer(depth=self.depth,
                                               dim=self.embed_dim,
                                               num_heads=self.num_heads,
                                               mlp_ratio=self.mlp_ratio,
                                               drop_rate=self.drop_rate)
        # Classifier
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        # Path Embeddings, CLS Token, Position Encoding
        x = self.patch_embeddings(x)
        # Transformer Encoder
        x = self.transformer_encoder(x)
        # Classifier
        #берем токен класса у всех батчей и их всех значений
        x = self.classifier(x[:, 0, :])
        return x
