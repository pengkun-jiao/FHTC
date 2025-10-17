
import torch
import torch.nn as nn
import re
from .ms_cross_attn import VCE_Module

import yaml
def load_yaml_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
ADAPTER_CONFIG = load_yaml_config('config/config.yaml')



class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    


class MMProjector(nn.Module):
    def __init__(self, config):
        super(MMProjector, self).__init__()  

        projector_type = getattr(config, 'mm_projector_type', 'linear')

        if projector_type == 'linear':
            self.mm_projector_llava = nn.Linear(config.mm_hidden_size, config.hidden_size)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.mm_projector_llava = nn.Sequential(*modules) 

        self.enable_VCE = ADAPTER_CONFIG.VCE.enable_VCE
        if self.enable_VCE:
            n_levels = ADAPTER_CONFIG.VCE.n_levels
            n_heads = ADAPTER_CONFIG.VCE.n_heads
            n_points = ADAPTER_CONFIG.VCE.n_points
            self.VCE_module = VCE_Module(d_model=config.mm_hidden_size, n_levels=n_levels, n_heads=n_heads, n_points=n_points)


    def forward(self, visual_feats, **kwargs):

        if  self.enable_VCE:
            visual_feats = self.VCE_module(visual_feats)
        
        x = self.mm_projector_llava(visual_feats)

        return x



def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
