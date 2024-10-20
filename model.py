import torch
import torch .nn as nn
from dataclasses import dataclass
import math
import torch.nn.functional as F


@dataclass
class VITConfig:
    n_embd: int = 768
    bias: bool = True
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    in_feature: int = 768
    out_featrue: int = 1000
    dropout: float = 0.0
    in_channel: int = 3
    block_size: int = 196
    patch_size: int = 14


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            config.in_channel, config.n_embd, kernel_size=16, stride=16)

    def forward(self, xb):
        xb = self.projection(xb)  # (B, n_embd, sH, sW)
        # (B, n_embd, sH, sW) -> (B, n_embd, sH*sW) -> (B, sh*sW, n_embd)
        xb = xb.flatten(2).transpose(1, 2)
        return xb


class ViTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = ViTPatchEmbeddings(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.position_embeddings = nn.Parameter(torch.zeros(
            1, config.patch_size * config.patch_size + 1, config.n_embd))
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, xb):
        B = xb.shape[0]
        xb = self.patch_embeddings(xb)  # (B, sH * sW, n_embd)
        # Add class tokens
        cls_token = self.cls_token.expand(
            B, -1, -1)  # (1, 1, 768) -> (B, 1, 768)
        # (B, sH * sW, n_embd) -> (B, sH * sW + 1, n_embd)
        xb = torch.cat((cls_token, xb), dim=1)
        xb = xb + self.position_embeddings
        return self.dropout(xb)


class VitSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.n_embd, config.n_embd, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, xb):
        return self.dropout(self.dense(xb))


class ViTSdpaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0, f"embedding sizea {config.n_embd} should be divisible by head_size {config.n_head}"
        self.query = nn.Linear(in_features=config.n_embd,
                               out_features=config.n_embd, bias=config.bias)
        self.key = nn.Linear(in_features=config.n_embd,
                             out_features=config.n_embd, bias=config.bias)
        self.value = nn.Linear(in_features=config.n_embd,
                               out_features=config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout()
        self.register_buffer('bias', torch.tril(torch.ones(
            config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, xb):
        # (B, sH * sW, n_embd) -> (B, sH * sW, n_embd) -> (B, sH * sW, nh, hs) -> (B, nh, sH * sW, hs)
        B, T, C = xb.shape
        q = self.query(xb).view(B, T, config.n_head, C //
                                config.n_head).transpose(1, 2)
        k = self.key(xb).view(B, T, config.n_head, C //
                              config.n_head).transpose(1, 2)
        v = self.value(xb).view(B, T, config.n_head, C //
                                config.n_head).transpose(1, 2)

        # (B, nh, sH * sW, hs) x (B, nH, hs, sH * sW) - (B, nH, sH * sW, sH * sW)
        attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(config.n_embd))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn))

        attn = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return attn


class ViTSdpAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = ViTSdpaSelfAttention(config)
        self.output = VitSelfOutput(config)

    def forward(self, xb):
        xb = self.attention(xb)
        return self.output(xb)


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(in_features=config.n_embd,
                               out_features=4 * config.n_embd, bias=config.bias)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, xb):
        return self.intermediate_act_fn(self.dense(xb))


class VitOutout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(in_features=4 * config.n_embd,
                               out_features=config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, xb):
        return self.dropout(self.dense(xb))


class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = ViTSdpAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = VitOutout(config)
        self.layernorm_before = nn.LayerNorm(config.n_embd)
        self.layernorm_after = nn.LayerNorm(config.n_embd)

    def forward(self, xb):
        xb = self.attention(xb)
        print("attn output", xb.shape)  # (B, sH*sW, n_embd)
        xb = self.intermediate(xb)
        print("internetidate output", xb.shape)  # (B, sH*sW, n_embd)
        xb = self.output(xb)
        print("output output", xb.shape)  # (B, sH*sW, n_embd)
        xb = self.layernorm_before(xb)
        print("layernomr before", xb.shape)  # (B, sH*sW, n_embd)
        xb = self.layernorm_after(xb)
        print("layernomr after", xb.shape)  # (B, sH*sW, n_embd)
        return xb


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config)
                                    for _ in range(config.n_layer)])

    def forward(self, xb):  # (B, n_embd, sH, sW)
        for layer in self.layers:
            xb = layer(xb)
        return xb


class VitModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbedding(self.config)
        self.encoder = ViTEncoder(self.config)
        self.layernorm = nn.LayerNorm(self.config.n_embd)

    def forward(self, xb):
        xb = self.embeddings(xb)  # (B, C, H, W) -> (B, n_embd, sH, sW)
        xb = self.encoder(xb)
        xb = self.layernorm(xb)
        return xb


class ViTImageClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vit = VitModel(config)
        self.classifier = nn.Linear(
            config.in_feature, config.out_featrue, bias=config.bias)

    def forward(self, xb):
        xb = self.vit(xb)  # (B, C, H, W) -> (B, config.n_embd)
        xb = self.classifier(xb)
        return xb

    @staticmethod
    def from_pretrained():
        from transformers import ViTForImageClassification
        model_hf = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float16)
        hf_sd = model_hf.state_dict()
        hf_sd_keys = hf_sd.keys()

        config = VITConfig()
        model = ViTImageClassifier(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attention.bias')]

        assert len(sd_keys) == len(hf_sd_keys), "keys length are not matching"
        for k in hf_sd_keys:
            assert hf_sd[k].shape == sd[k].shape, f"not matching {k}"
            with torch.no_grad():
                sd[k].copy_(hf_sd[k])
        return model


config = VITConfig()
# load huggingface model weight to verify
model = ViTImageClassifier.from_pretrained()
print("model loaded")
