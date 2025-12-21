import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ============================================================
# nnU-Netv2 스타일 2D 구성 요소
# ============================================================

class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act  = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, k=3, s=1, p=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        out = self.act(out + identity)
        return out

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2, downsample=True):
        super().__init__()
        self.down = None
        if downsample:
            self.down = ConvNormAct(in_ch, out_ch, k=3, s=2, p=1, bias=False)
            blk_in = out_ch
        else:
            self.down = ConvNormAct(in_ch, out_ch, k=3, s=1, p=1, bias=False)
            blk_in = out_ch

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(blk_in, out_ch))
            blk_in = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x

class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.up_norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.up_act  = nn.LeakyReLU(0.01, inplace=True)

        self.reduce = ConvNormAct(out_ch + skip_ch, out_ch, k=3, s=1, p=1, bias=False)

        blocks = []
        blk_in = out_ch
        for _ in range(n_blocks):
            blocks.append(ResidualBlock(blk_in, out_ch))
            blk_in = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.up_act(self.up_norm(x))

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.blocks(x)
        return x

def _make_feature_list(base, num_stages, max_f):
    feats = []
    f = base
    for _ in range(num_stages):
        feats.append(int(min(f, max_f)))
        f *= 2
    return feats

class nnUNetV2_2D(nn.Module):
    def __init__(self, in_ch=3, num_classes=29,
                 base_features=32,
                 num_stages=5,
                 max_features=320,
                 n_blocks_per_stage=2,
                 deep_supervision=True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        feats = _make_feature_list(base_features, num_stages, max_features)

        self.enc_stages = nn.ModuleList()
        for i in range(num_stages):
            in_c = in_ch if i == 0 else feats[i-1]
            out_c = feats[i]
            downsample = (i != 0)
            self.enc_stages.append(EncoderStage(in_c, out_c, n_blocks=n_blocks_per_stage, downsample=downsample))

        self.dec_stages = nn.ModuleList()
        cur_ch = feats[-1]
        for level in range(num_stages - 2, -1, -1):
            skip_ch = feats[level]
            out_ch  = feats[level]
            self.dec_stages.append(DecoderStage(cur_ch, skip_ch, out_ch, n_blocks=n_blocks_per_stage))
            cur_ch = out_ch

        self.head_main = nn.Conv2d(feats[0], num_classes, kernel_size=1, bias=True)

        self.aux_heads = nn.ModuleList()
        if self.deep_supervision:
            for ch in feats[1: min(4, len(feats))]:
                self.aux_heads.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        skips = []
        for i, st in enumerate(self.enc_stages):
            x = st(x)
            skips.append(x)

        x = skips[-1]

        dec_feats = []
        for di, dec in enumerate(self.dec_stages):
            skip = skips[-2 - di]
            x = dec(x, skip)
            dec_feats.append(x)

        full_feat = dec_feats[-1]
        out_main = self.head_main(full_feat)

        aux_outs = []
        if self.deep_supervision:
            aux_feats = []
            if len(dec_feats) >= 2: aux_feats.append(dec_feats[-2])
            if len(dec_feats) >= 3: aux_feats.append(dec_feats[-3])
            if len(dec_feats) >= 4: aux_feats.append(dec_feats[-4])

            for head, feat in zip(self.aux_heads, aux_feats):
                aux_outs.append(head(feat))

        return {"out": out_main, "aux": aux_outs}

# ============================================================
# train.py 호환을 위한 get_model 함수
# ============================================================

def get_model():
    """
    work/train.py에서 호출할 수 있는 인터페이스입니다.
    Config에 정의된 클래스 개수를 자동으로 연동합니다.
    """
    model = nnUNetV2_2D(
        in_ch=3,
        num_classes=len(Config.CLASSES),
        base_features=32,
        num_stages=5,
        deep_supervision=True  # train.py에서 딕셔너리 출력을 처리하므로 True 가능
    )
    return model
