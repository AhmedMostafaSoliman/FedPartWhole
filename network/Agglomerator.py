import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pytorch_lightning as pl
from utils.ccutils import exists, default, Siren
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

TOKEN_ATTEND_SELF_VALUE = -5e-4

class ConvTokenizer(pl.LightningModule):
    def __init__(self, in_channels=3, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)


class ColumnNet(pl.LightningModule):
    def __init__(self, FLAGS, dim, groups, mult = 4, activation = nn.GELU):
        super().__init__()
        self.FLAGS = FLAGS
        total_dim = dim * groups
        num_patches = (self.FLAGS.conv_image_size // self.FLAGS.patch_size) ** 2
        
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups = groups),
            activation(),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim * mult, total_dim, 1, groups = groups),
            Rearrange('b (l d) n -> b n l d', l = groups)
        )

    def forward(self, levels):
        levels = self.net(levels)
        return levels
    

class ColumnEncoder(pl.LightningModule):
    def __init__(self, FLAGS, dim, groups, mult = 4, out_embed_dim=256, activation = nn.GELU):
        super().__init__()
        self.FLAGS = FLAGS
        total_dim = dim * groups
        num_patches = (self.FLAGS.conv_image_size // self.FLAGS.patch_size) ** 2
        
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups = groups),
            activation(),
            nn.LayerNorm(num_patches),
            nn.Conv1d(total_dim * mult, out_embed_dim*groups, 1, groups = groups),
            Rearrange('b (l d) n -> b n l d', l = groups)
        )

    def forward(self, levels):
        levels = self.net(levels)
        return levels

class ConsensusAttention(pl.LightningModule):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out

class Agglomerator(pl.LightningModule):
    def __init__(self,
        FLAGS,
        *,
        consensus_self = False,
        local_consensus_radius = 0
        ):
        super(Agglomerator, self).__init__()
        self.FLAGS = FLAGS

        self.num_patches_side = (self.FLAGS.conv_image_size // self.FLAGS.patch_size)
        self.num_patches =  self.num_patches_side ** 2
        self.features = []
        self.labels = []
        self.iters = default(self.FLAGS.iters, self.FLAGS.levels * 2)
        self.batch_acc = 0


        self.wl = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wBU = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wTD = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)
        self.wA = torch.nn.parameter.Parameter(torch.tensor(0.25, device=self.device), requires_grad=True)

        self.image_to_tokens = nn.Sequential(
            ConvTokenizer(in_channels=self.FLAGS.n_channels, embedding_dim=self.FLAGS.patch_dim // (self.FLAGS.patch_size ** 2)),
            Rearrange('b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1 = self.FLAGS.patch_size, p2 = self.FLAGS.patch_size),
        )

        self.classification_head_from_last_level = nn.Sequential(
            nn.LayerNorm(FLAGS.patch_dim),
            nn.Dropout(p=0.5),
            Rearrange('b n d -> b (n d)'),
            nn.LayerNorm(self.num_patches * FLAGS.patch_dim),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_patches * FLAGS.patch_dim, FLAGS.n_classes),
        )

        self.init_levels = nn.Parameter(torch.randn(self.FLAGS.levels, FLAGS.patch_dim))
        self.bottom_up = ColumnNet(self.FLAGS, dim = FLAGS.patch_dim, activation=nn.GELU, groups = self.FLAGS.levels)
        self.top_down = ColumnNet(self.FLAGS, dim = FLAGS.patch_dim, activation=Siren, groups = self.FLAGS.levels - 1)
       
        self.attention = ConsensusAttention(self.num_patches_side, attend_self = consensus_self, local_consensus_radius = local_consensus_radius)
        if self.FLAGS.levels_encoder:
            print('Using the encoder')
            self.levels_encoder = ColumnEncoder(self.FLAGS, dim = FLAGS.patch_dim, activation=nn.GELU, groups = self.FLAGS.levels)
        else:
            print('Not using the encoder')
        
    def forward(self, img, levels = None):
        b, device = img.shape[0], img.device

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        hiddens = [levels]

        num_contributions = torch.empty(self.FLAGS.levels, device = device).fill_(4)
        num_contributions[-1] = 3 

        for _ in range(self.iters):
            levels_with_input = torch.cat((bottom_level, levels), dim = -2)
            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])
            top_down_out = self.top_down(torch.flip(levels_with_input[..., 2:, :], [2]))
            top_down_out = F.pad(torch.flip(top_down_out, [2]), (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels)

            levels_sum = torch.stack((
                levels * self.wl, \
                bottom_up_out * self.wBU, \
                top_down_out * self.wTD, \
                consensus * self.wA
            )).sum(dim = 0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            self.log('Weights/wl', self.wl)
            self.log('Weights/wBU', self.wBU)
            self.log('Weights/wTD', self.wTD)
            self.log('Weights/wA', self.wA)

            levels = levels_mean
            hiddens.append(levels)

        all_levels = torch.stack(hiddens)

        top_level = all_levels[self.FLAGS.denoise_iter, :, :, -1]

        top_level = F.normalize(top_level, dim=1)
        encoded_levels_out = all_levels[-1, :, :, :, :]
        if self.FLAGS.levels_encoder:
            encoded_levels_out = self.levels_encoder(all_levels[-1, :, :, :, :])
        if self.FLAGS.supervise:
            out = self.classification_head_from_last_level(top_level)
            return out, encoded_levels_out
        else:
            return top_level, all_levels[-1,0,:,:,:], encoded_levels_out