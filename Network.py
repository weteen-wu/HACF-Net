import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SOCAB_Attention(nn.Module):
    # overlapping cross-attention block
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm,

                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(self.window_size * self.overlap_ratio) + self.window_size


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH


        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        self.relative_position_index = relative_coords.sum(-1)


        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, mask=None):
        b, h, w, c = x.shape
        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1,
                                                                 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # shape (nw*b, nH,8*8,12*12)
        B_, nH, N1, N2 = attn.shape
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size,
            -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # (nW, window_size*window_size, window_size*window_size),nW是num_windows
            attn = attn.view(B_ // nW, nW, nH, N1, N2) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, nH, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        attn_windows = self.proj(attn_windows)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c

        return shifted_x


class SOCAB_TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SOCAB_TransformerBlock, self).__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.overlap_win_size = int(window_size * 0.5) + window_size

        if min(self.input_resolution) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        self.norm1 = norm_layer(dim)
        self.attn = SOCAB_Attention(dim, input_resolution=input_resolution,
                                   window_size=self.window_size, overlap_ratio=0.5, num_heads=num_heads,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, mlp_ratio=2, norm_layer=norm_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # dim=60,mlp_ratio=2.
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def calculate_mask(self, x_size):

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1)).cuda()
        B, H, W, C = img_mask.shape
        h_slice1 = (slice(self.shift_size, None),
                    slice(0, self.shift_size),
                    )
        w_slice1 = (slice(self.shift_size, None),
                    )

        w_slice2 = (slice(0, self.shift_size),)
        h_slice2 = (slice(self.shift_size, None),
                    slice(0, self.shift_size),
                    )

        cnt = 1
        for h in h_slice1:
            for w in w_slice1:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        for w in w_slice2:
            for h in h_slice2:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (9,4)

        mask_windows_large = self.unfold(img_mask.view(B, C, H, W))
        # (B,window_size*window_size,patch)
        mask_windows_large = rearrange(mask_windows_large, 'b ( ch owh oww) nw -> (b nw) (owh oww) ch',
                                       ch=C, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        _, L, _ = mask_windows_large.shape
        mask_windows_large = mask_windows_large.view(-1, L)
        attn_mask = mask_windows.unsqueeze(2) - mask_windows_large.unsqueeze(
            1)  #(nW,window_size*window_size,window_size*window_size)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        if self.input_resolution == x_size:
            shifted_x = self.attn(shifted_x, mask=self.attn_mask)  # (nW*B, window_size*window_size, C)
        else:
            shifted_x = self.attn(shifted_x, mask=self.calculate_mask(x_size).to(x.device))

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OCA_BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.
                 , qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super(OCA_BasicLayer, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SOCAB_TransformerBlock(dim=dim, input_resolution=input_resolution,
                                       num_heads=num_heads, window_size=window_size,
                                       shift_size=0 if (i % 2 == 0) else window_size // 2,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                       norm_layer=norm_layer
                                       )
            for i in range(depth)
        ])
        # patch merging layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution=input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.upsample is not None:
            x, x_size = self.upsample(x, x_size)
        return x, x_size


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)  ## 2*Wh-1 * 2*Ww-1, nH
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2,Wh,Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)  # (3,B_,self.num_heads,N,C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_,self.num_heads,N,N)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]  # (nW, window_size*window_size, window_size*window_size)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # attn@v -->(B_,self.num_heads,N,C // self.num_heads)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # dim=60,mlp_ratio=2.
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.attn_mask = self.calculate_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1)).cuda()
        h_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slice:
            for w in w_slice:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2)  # (nW,window_size*window_size,window_size*window_size)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, window_size*window_size, C)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  ## B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.
                 , qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super(BasicLayer, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer
                                 )
            for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x, x_size = self.downsample(x, x_size)
        return x, x_size


class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         )

    def forward(self, x, x_size):
        return self.residual_group(x, x_size)


class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

    def forward(self, x, x_size):
        """
        x: B, L = H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))
        out = out.permute(0, 2, 3, 1)  # B, H, W, C
        x_size = (out.shape[1], out.shape[2])
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        return out, x_size


class PatchMerging_normal(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x_size = (x.shape[1], x.shape[2])
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x, x_size



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patch_resolution
        self.num_patches = patch_resolution[0] * patch_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchUnEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1]).contiguous()
        return x

class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=k_size, padding=1, bias=True),
            nn.ReLU()
        )
        self.k3 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        shortcut = self.k2(x)
        y = self.k3(shortcut)
        y = self.sigmoid(y)

        out = torch.mul(self.k4(shortcut), y) + x

        return out


class Fusion_Block(nn.Module):
    def __init__(self, embed_dim, patches_resolution, window_size, depths, num_heads
                 , mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint,
                 img_size, patch_size):
        super(Fusion_Block, self).__init__()
        self.norm = norm_layer
        self.num_features = embed_dim * 2
        self.patch_norm = True
        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim

        self.patch_embed_16 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim // 2, embed_dim=embed_dim // 2,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_unembed_32 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        self.patch_embed_32 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patch_unembed_64 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None
        )

        self.patch_embed_64 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 2, embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patch_unembed_256 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 8, embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patch_embed_128 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 4, embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patch_unembed_128 = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim * 4, embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.norm_first = self.norm(self.num_features)

        self.layer_1 = RSTB(dim=embed_dim // 2,
                            input_resolution=(patches_resolution // (2 ** 0),
                                              patches_resolution // (2 ** 0)),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )
        self.layer1 = OCA_BasicLayer(
            dim=embed_dim,
            input_resolution=(patches_resolution // (2 ** 0),
                              patches_resolution // (2 ** 0)),
            window_size=window_size,
            depth=depths[1],
            num_heads=num_heads[1],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
            norm_layer=norm_layer,
            upsample=None,
            use_checkpoint=use_checkpoint, )
        self.layer_2 = RSTB(dim=embed_dim,
                            input_resolution=(patches_resolution // (2 ** 1),
                                              patches_resolution // (2 ** 1)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )
        self.layer2 = OCA_BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution // (2 ** 1),
                              patches_resolution // (2 ** 1)),
            window_size=window_size,
            depth=depths[3],
            num_heads=num_heads[3],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
            norm_layer=norm_layer,
            upsample=None,
            use_checkpoint=use_checkpoint, )

        self.layer_3 = RSTB(dim=embed_dim * 2,
                            input_resolution=(patches_resolution // (2 ** 2),
                                              patches_resolution // (2 ** 2)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )

        self.layer3 = OCA_BasicLayer(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution // (2 ** 2),
                              patches_resolution // (2 ** 2)),
            window_size=window_size,
            depth=depths[5],
            num_heads=num_heads[5],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
            norm_layer=norm_layer,
            upsample=None,
            use_checkpoint=use_checkpoint, )

        self.layer4 = OCA_BasicLayer(
            dim=embed_dim * 8,
            input_resolution=(patches_resolution // (2 ** 3),
                              patches_resolution // (2 ** 3)),
            window_size=window_size,
            depth=depths[6],
            num_heads=num_heads[6],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:6]):sum(depths[:7])],
            norm_layer=norm_layer,
            upsample=None,
            use_checkpoint=use_checkpoint, )

        self.upscale = UpSample(input_resolution=patches_resolution // 8, in_channels=embed_dim * 8, scale_factor=2)

        self.layer_5 = RSTB(dim=embed_dim * 2,
                            input_resolution=(patches_resolution // (2 ** 2),
                                              patches_resolution // (2 ** 2)),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:7]):sum(depths[:8])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )

        self.layer5 = OCA_BasicLayer(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution // (2 ** 2),
                              patches_resolution // (2 ** 2)),
            window_size=window_size,
            depth=depths[8],
            num_heads=num_heads[8],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:8]):sum(depths[:9])],
            norm_layer=norm_layer,
            upsample=UpSample,
            use_checkpoint=use_checkpoint, )

        self.layer_6 = RSTB(dim=embed_dim,
                            input_resolution=(patches_resolution // (2 ** 1),
                                              patches_resolution // (2 ** 1)),
                            depth=depths[9],
                            num_heads=num_heads[9],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:9]):sum(depths[:10])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )
        self.layer6 = OCA_BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution // (2 ** 1),
                              patches_resolution // (2 ** 1)),
            window_size=window_size,
            depth=depths[10],
            num_heads=num_heads[10],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:10]):sum(depths[:11])],
            norm_layer=norm_layer,
            upsample=UpSample,
            use_checkpoint=use_checkpoint, )

        self.layer_7 = RSTB(dim=embed_dim // 2,
                            input_resolution=(patches_resolution // (2 ** 0),
                                              patches_resolution // (2 ** 0)),
                            depth=depths[11],
                            num_heads=num_heads[11],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:11]):sum(depths[:12])],
                            norm_layer=norm_layer,

                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size, )
        self.layer7 = OCA_BasicLayer(
            dim=embed_dim,
            input_resolution=(patches_resolution // (2 ** 0),
                              patches_resolution // (2 ** 0)),
            window_size=window_size,
            depth=depths[12],
            num_heads=num_heads[12],
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:12]):sum(depths[:13])],
            norm_layer=norm_layer,
            upsample=None,
            use_checkpoint=use_checkpoint, )

        self.split_channel1_1 = PAConv(16)
        self.split_channel1_2 = PAConv(16)

        self.split_channel2_1 = PAConv(32)
        self.split_channel2_2 = PAConv(32)

        self.split_channel3_1 = PAConv(64)
        self.split_channel3_2 = PAConv(64)

        self.conv_concat1 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_concat_last = nn.Conv2d(64, 64, kernel_size=1)
        self.upscale_concat1 = UpSample(input_resolution=patches_resolution // 4,
                                        in_channels=embed_dim * 4, scale_factor=2)

        self.upscale_concat2 = UpSample(input_resolution=patches_resolution // 2,
                                        in_channels=embed_dim * 2, scale_factor=2)
        self.conv_concat2 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv_concat_last2 = nn.Conv2d(32, 32, kernel_size=1)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        self.downsample1 = PatchMerging_normal(input_resolution=patches_resolution, dim=embed_dim)
        self.downsample2 = PatchMerging_normal(input_resolution=patches_resolution // 2, dim=embed_dim * 2)
        self.downsample3 = PatchMerging_normal(input_resolution=patches_resolution // 4, dim=embed_dim * 4)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x, x_size):
        x_a, x_b = torch.split(x, (16, 16), dim=1)
        x_a_1 = self.split_channel1_1(x_a)
        x_a_1 = self.split_channel1_2(x_a_1)
        x_b, _ = self.layer_1(self.patch_embed_16(x_b), x_size)
        x1 = self.patch_unembed_32(torch.cat((self.patch_embed_16(x_a_1), x_b), dim=-1), x_size)
        x1_1, x_size_1 = self.layer1(self.patch_embed_32(self.channel_shuffle(x1, 8)), x_size)
        x1, x_size1 = self.downsample1(x1_1, x_size_1)  # (B,64,128,128)


        x_a1, x_b1 = torch.split(self.patch_unembed_64(x1, x_size1), (32, 32), dim=1)
        x_a1_1 = self.split_channel2_1(x_a1)
        x_a1_1 = self.split_channel2_2(x_a1_1)
        x_b1, _ = self.layer_2(self.patch_embed_32(x_b1), x_size1)
        x2 = self.patch_unembed_64(torch.cat((self.patch_embed_32(x_a1_1), x_b1), dim=-1), x_size1)
        x2_2, x_size_2 = self.layer2(self.patch_embed_64(self.channel_shuffle(x2, 16)), x_size1)
        x2, x_size2 = self.downsample2(x2_2, x_size_2)


        x_a2, x_b2 = torch.split(self.patch_unembed_128(x2, x_size2), (64, 64), dim=1)
        x_a2_2 = self.split_channel3_1(x_a2)
        x_a2_2 = self.split_channel3_2(x_a2_2)
        x_b2, _ = self.layer_3(self.patch_embed_64(x_b2), x_size2)
        x3 = self.patch_unembed_128(torch.cat((self.patch_embed_64(x_a2_2), x_b2), dim=-1), x_size2)
        x3_3, x_size_3 = self.layer3(self.patch_embed_128(self.channel_shuffle(x3, 32)), x_size2)
        x3, x_size3 = self.downsample3(x3_3, x_size_3)

        x4, x_size4 = self.layer4(x3, x_size3)
        x5, x_size5 = self.upscale(x4, x_size4)


        x6 = self.conv1_1(self.patch_unembed_256(torch.cat((x5, x3_3), dim=-1), x_size5))
        x_a3, x_b3 = torch.split(x6, (64, 64), dim=1)
        x_a3_1 = self.split_channel3_1(x_a3)
        x_a3_1 = self.split_channel3_2(x_a3_1)
        x_b3, _ = self.layer_5(self.patch_embed_64(x_b3), x_size5)
        x6 = self.patch_unembed_128(torch.cat((self.patch_embed_64(x_a3_1), x_b3), dim=-1), x_size5)
        x6, x_size6 = self.layer5(self.patch_embed_128(self.channel_shuffle(x6, 32)), x_size5)


        x_low3 = self.conv_concat1(self.patch_unembed_64(self.upscale_concat1(x3_3, x_size_3)[0], x_size_2))
        x2_2_new = self.conv_concat1(self.patch_unembed_64(x2_2, x_size_2))
        x2_2_new = self.conv_concat_last(torch.cat((x_low3, x2_2_new), dim=1))
        x7 = self.conv1_2(torch.cat((self.patch_unembed_64(x6, x_size6), x2_2_new), dim=1))
        x_a4, x_b4 = torch.split(x7, (32, 32), dim=1)
        x_a4_1 = self.split_channel2_1(x_a4)
        x_a4_1 = self.split_channel2_2(x_a4_1)
        x_b4, _ = self.layer_6(self.patch_embed_32(x_b4), x_size6)
        x7 = self.patch_unembed_64(torch.cat((self.patch_embed_32(x_a4_1), x_b4), dim=-1), x_size6)
        x7, x_size7 = self.layer6(self.patch_embed_64(self.channel_shuffle(x7, 16)), x_size6)


        x_low2 = self.conv_concat2(
            self.patch_unembed_32(self.upscale_concat2(self.patch_embed_64(x2_2_new), x_size_2)[0], x_size))
        x1_1_new = self.conv_concat2(self.patch_unembed_32(x1_1, x_size_1))
        x1_1_new = self.conv_concat_last2(torch.cat((x_low2, x1_1_new), dim=1))
        x8 = self.conv1_3(torch.cat((self.patch_unembed_32(x7, x_size7), x1_1_new), dim=1))
        x_a5, x_b5 = torch.split(x8, (16, 16), dim=1)
        x_a5_1 = self.split_channel1_1(x_a5)
        x_a5_1 = self.split_channel1_2(x_a5_1)
        x_b5, _ = self.layer_7(self.patch_embed_16(x_b5), x_size7)
        x8 = self.patch_unembed_32(torch.cat((self.patch_embed_16(x_a5_1), x_b5), dim=-1), x_size7)
        x8, x_size8 = self.layer7(self.patch_embed_32(self.channel_shuffle(x8, 8)), x_size7)
        x8 = self.patch_unembed_32(x8, x_size8)

        return x8 + x, x_size8


class ResLayerPool(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResLayerPool, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=2, bias=False)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        res = self.pool(x)
        return out + res


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_size):
        b, N, c = x.shape
        h, w = x_size
        x = x.view(b, c, h, w)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Transposed_Attention_initial(nn.Module):
    def __init__(self, dim1, dim2, num_heads, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Transposed_Attention_initial, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim1, 2 * dim2, kernel_size=1, stride=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim2, dim2, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x, x_size):
        b, N, c = x.shape
        h, w = x_size
        x = x.view(b, c, h, w)
        qkv = self.qkv(x)
        q, v = qkv.chunk(2, dim=1)

        k = self.kv_dwconv(q)
        v = self.kv_dwconv(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        out = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return out


class TransposedBlock_initial(nn.Module):
    def __init__(self, dim1, dim2, num_heads, qkv_bias=False, act_layer=nn.GELU, mlp_ratio=4., drop=0.):
        super(TransposedBlock_initial, self).__init__()
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim2)
        self.ffn = FeedForward(dim=dim2, ffn_expansion_factor=2, bias=False)
        self.Transposed_Attention_initial = Transposed_Attention_initial(dim1, dim2, num_heads)
        self.conv1 = nn.Conv2d(dim1, dim2, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape
        x_size = (x.shape[2], x.shape[3])
        x = x.view(b, h * w, c)
        x = self.conv1(shortcut) + self.Transposed_Attention_initial(self.norm1(x), x_size)
        c = x.shape[1]
        x_norm = x.view(b, h * w, c)

        x = x + self.ffn(self.norm2(x_norm), x_size)
        x = x.view(b, c, h, w)
        return x


class HACF_Net(nn.Module):
    def __init__(self, sensing_rate=0.125, img_size=256, patch_size=1, in_chans=1,
                 embed_dim=32, depths=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                 num_heads=[4, 8, 4, 8, 4, 8, 8, 4, 8, 4, 8, 4, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2):
        super(HACF_Net, self).__init__()
        self.sensing_rate = sensing_rate
        self.base = 32
        self.embed_dim = embed_dim
        self.num_in_ch = in_chans

        if sensing_rate == 0.015625:

            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,16,16)
                nn.Conv2d(self.embed_dim, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 16
            self.initial = TransposedBlock_initial(dim1=4, dim2=256, num_heads=8)

        elif sensing_rate == 0.03125:
            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,32,32)
                nn.Conv2d(self.embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 8
            self.initial = TransposedBlock_initial(dim1=2, dim2=64, num_heads=8)

        elif sensing_rate == 0.0625:
            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,64,64)
                nn.Conv2d(self.embed_dim, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 8
            self.initial = TransposedBlock_initial(dim1=4, dim2=64, num_heads=8)

        elif sensing_rate == 0.125:
            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,32,32)
                nn.Conv2d(self.embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 4
            self.initial = TransposedBlock_initial(dim1=2, dim2=16, num_heads=8)

        elif sensing_rate == 0.25:
            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,64,64)
                nn.Conv2d(self.embed_dim, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 4
            self.initial = TransposedBlock_initial(dim1=4, dim2=16, num_heads=8)

        elif sensing_rate == 0.5:
            self.sampling = nn.Sequential(
                nn.Conv2d(1, self.embed_dim, kernel_size=3, padding=1, stride=1, bias=False),
                ResLayerPool(self.embed_dim, self.embed_dim),  # (1,32,128,128)
                nn.Conv2d(self.embed_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.m = 2
            self.initial = TransposedBlock_initial(dim1=2, dim2=4, num_heads=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.num_in_ch, embed_dim, 3, 1, 1),  # (B,32,256,256)
            nn.ReLU()
        )

        self.deep_rec = Fusion_Block(embed_dim=embed_dim, patches_resolution=256, window_size=window_size,
                                     depths=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                     drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=ape,
                                     patch_norm=patch_norm,
                                     use_checkpoint=use_checkpoint, img_size=img_size, patch_size=patch_size)

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.base, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=True)
        )

    def forward(self, x):
        xsample = self.sampling(x)
        x = self.initial(xsample)
        initial = nn.PixelShuffle(self.m)(x)

        x = self.conv1(initial)
        x_size = (x.shape[2], x.shape[3])
        x, x_size = self.deep_rec(x, x_size)
        out = self.conv2(x)

        return initial + out, initial

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input = torch.randn(1,1,256,256)
#     input = input.to(device)
#     net = HACF_Net(sensing_rate=0.5, img_size=256, window_size=8,
#                     depths=[4,4,4,4,4,4,4,4,4,4,4,4,4],num_heads=[4,8,4,8,4,8,8,4,8,4,8,4,8], mlp_ratio=4.)
#     net = net.to(device)

#     output = net(input)
#     print(output[0].size())





