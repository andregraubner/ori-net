from plasmamba import CaduceusForMaskedLM, CaduceusMixerModel, CaduceusConfig
import torch
from torch import nn
from torch import nn
from transformers import AutoModel, AutoTokenizer
from plasmamba import CaduceusForMaskedLM, CaduceusConfig
import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from TorchCRF import CRF
import math
from rectified_flow import RectifiedFlow
from einops import repeat, rearrange
from plashydra import Hydra

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def exists(x):
    return x is not None

flow_config = CaduceusConfig(
    d_model = 512,
    n_layer = 1,
    #vocab_size = tokenizer.vocab_size,
    #pad_token_id = tokenizer.pad_token_id,

    # Caduceus-specific params
    bidirectional = True,
    bidirectional_strategy = "add",
    bidirectional_weight_tie = True,
    rcps = False,
    use_mamba1=False
)

config = CaduceusConfig(
    d_model = 512,
    n_layer = 8,
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_token_id,

    # Caduceus-specific params
    bidirectional = True,
    bidirectional_strategy = "add",
    bidirectional_weight_tie = True,
    rcps = False,
    use_mamba1=False
)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

"""
class OriFlow(nn.Module):
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.layers = nn.ModuleList([])

        for i in range(1):
            self.layers.append(nn.ModuleList([
                    CaduceusMixerModel(flow_config), 
                    nn.Linear(512, 256)
                ]))

        #self.up_proj = nn.Linear(1,256)
        self.init_vec = nn.Parameter(torch.rand(256))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

    def forward(self, noised, times, cond):

        cond = self.plasmamba(cond, output_hidden_states=True)["hidden_states"][-1]

        b, s = noised.shape[:2]
        t = self.time_mlp(times).unsqueeze(1)
        #z = torch.zeros(b, s, 255, device=noised.device)
        #z = repeat(self.init_vec, "d -> b s d", b=b, s=s)
        noised = self.up_proj(noised)
        #noised = torch.cat([z, noised], dim=2)

        for mamba, down_proj in self.layers:

            c = down_proj(cond)
            x = torch.cat([noised, c], dim=2)
            x = torch.cat([t, x, t], dim=1)
            x, _ = mamba(input_ids=None, inputs_embeds=x)
            x = x[:,1:-1,:256]

        x = x[:,:,:1]
        
        return x


class OriFlow(nn.Module):
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.layers = nn.ModuleList([])

        for i in range(4):
            self.layers.append(nn.ModuleList([
                    CaduceusMixerModel(flow_config), 
                    nn.Linear(512, 1024)
                ]))

        self.up_proj = nn.Linear(1, 512)
        self.init_vec = nn.Parameter(torch.rand(256))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.init_conv = nn.Conv1d(1, 512, 3, padding='same', padding_mode="circular")
        self.final_conv = nn.Conv1d(512, 1, 3, padding='same', padding_mode="circular")

    def forward(self, noised, times, cond):

        cond = self.plasmamba(cond, output_hidden_states=True)["hidden_states"][-1]

        b, s = noised.shape[:2]
        t = self.time_mlp(times).unsqueeze(1)
        #z = torch.zeros(b, s, 255, device=noised.device)
        #z = repeat(self.init_vec, "d -> b s d", b=b, s=s)
        noised = rearrange(noised, "b s c -> b c s")
        noised = self.init_conv(noised)
        noised = rearrange(noised, "b c s -> b s c")
        #noised = torch.cat([z, noised], dim=2)
        
        for mamba, down_proj in self.layers:

            c = down_proj(cond)
            scale, shift = c.chunk(2, dim = 2)
            x = noised * (scale + 1) + shift
            x = torch.cat([t, x, t], dim=1)
            x, _ = mamba(input_ids=None, inputs_embeds=x)
            x = x[:,1:-1]

        #x = x[:,:,:1]
        x = rearrange(x, "b s c -> b c s")
        x = self.final_conv(x)
        x = rearrange(x, "b c s -> b s c")
        
        return x
"""

class Block(nn.Module):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, 3, padding='same', padding_mode="circular")
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, cond_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )

        self.block1 = Block(dim, groups = groups)
        self.block2 = Block(dim, groups = groups)

    def forward(self, x, cond):

        cond = self.mlp(cond)
        cond = rearrange(cond, "b s c -> b c s")
        cond = cond.chunk(2, dim = 1)
        h = self.block1(x, scale_shift=cond)
        h = self.block2(h)

        return h + x

class HydraBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = Hydra(d_model=dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        
        x = self.proj(x)
        #x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class HydraLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.block1 = HydraBlock(dim)
        self.block2 = HydraBlock(dim)

    def forward(self, x, cond):

        cond = self.mlp(cond)
        cond = cond.chunk(2, dim = -1)
        
        x = self.norm1(x)
        h = self.block1(x, scale_shift=cond)
        
        x = self.norm2(h)
        h = self.block2(h)

        return h + x

class OriFlow(nn.Module):
    def __init__(self):
        super().__init__()
        model = CaduceusForMaskedLM(config)
        model.load_state_dict(torch.load("/root/autodl-fs/weights/model_9.pth"))
        self.plasmamba = model
        self.encoder = nn.ModuleList([HydraLayer(dim=256, cond_dim=512) for i in range(4)])
        self.decoder = nn.ModuleList([HydraLayer(dim=256, cond_dim=256) for i in range(4)])

        self.down_proj = nn.Linear(512, 256)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

        self.init_conv = nn.Conv1d(1, 256, 3, padding='same', padding_mode="circular")
        self.final_conv = nn.Conv1d(256, 1, 3, padding='same', padding_mode="circular")

    def forward(self, noised, times, cond):

        cond = self.plasmamba(cond, output_hidden_states=True)["hidden_states"][-1]

        b, s = noised.shape[:2]
        t = self.time_mlp(times)
        
        x = rearrange(noised, "b s c -> b c s")
        x = self.init_conv(x)
        x = rearrange(x, "b c s -> b s c")

        c = self.down_proj(cond)
        
        for conditioning in self.encoder:
            residual = c
            c = conditioning(c, t)
            c += residual

        for hydra in self.decoder:
            residual = x
            x = hydra(x, c)
            x += residual

        x = rearrange(x, "b s c -> b c s")
        x = self.final_conv(x)
        x = rearrange(x, "b c s -> b s c")
        
        return x