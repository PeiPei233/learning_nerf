import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class NeRF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        xyz_input_ch = kwargs['xyz_input_ch']
        dir_input_ch = kwargs['dir_input_ch']
        D, W  = kwargs['D'], kwargs['W']
        skips = kwargs['skips']
        V_D = kwargs['V_D']
        self.pts_layers = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W)] + 
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + xyz_input_ch, W) for i in range(D-1)]
        )

        self.dir_layers = nn.ModuleList(
            [nn.Linear(dir_input_ch + W, W // 2)] + 
            [nn.Linear(W // 2, W // 2) for i in range(V_D-1)]
        )

        self.feature_layer = nn.Linear(W, W)
        self.sig_layer = nn.Linear(W, 1)
        self.rgb_layer = nn.Linear(W // 2, 3)

        self.skips = skips
        self.D = D
        self.W = W
        self.xyz_input_ch = xyz_input_ch
        self.dir_input_ch = dir_input_ch
    
    def forward(self, x):
        xyz_input = x['xyz']
        dir_input = x['dir']
        xyz = xyz_input
        for i, l in enumerate(self.pts_layers):
            xyz = self.pts_layers[i](xyz)
            xyz = F.relu(xyz)
            if i in self.skips:
                xyz = torch.cat([xyz, xyz_input], dim=-1)
        sig_output = F.relu(self.sig_layer(xyz))
        xyz = self.feature_layer(xyz)
        xyz = torch.cat([xyz, dir_input], dim=-1)
        for i, l in enumerate(self.dir_layers):
            xyz = self.dir_layers[i](xyz)
            xyz = F.relu(xyz)
        rgb_output = torch.sigmoid(self.rgb_layer(xyz))
        return {'rgb': rgb_output, 'sig': sig_output}



class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.xyz_encoder, self.xyz_input_ch = get_encoder(cfg.network.xyz_encoder)
        self.dir_encoder, self.dir_input_ch = get_encoder(cfg.network.dir_encoder)

        net_cfg = {
            'xyz_input_ch': self.xyz_input_ch,
            'dir_input_ch': self.dir_input_ch,
            'D': cfg.network.nerf.D,
            'W': cfg.network.nerf.W,
            'skips': cfg.network.nerf.skips,
            'V_D': cfg.network.nerf.V_D
        }

        self.nerf = NeRF(**net_cfg)
        self.fine_nerf = NeRF(**net_cfg)

        self.near = 2.
        self.far = 6.
        self.N_c, self.N_f = cfg.task_arg.cascade_samples
        self.chunk = cfg.task_arg.chunk_size
    
    def batchify(self, model, x):
        xyz = x['xyz']
        dir = x['dir']
        N = xyz.shape[0]
        ret = {}
        for i in range(0, N, self.chunk):
            x = model({'xyz': xyz[i:i+self.chunk], 'dir': dir[i:i+self.chunk]})
            if i == 0:
                for k in x:
                    ret[k] = x[k]
            else:
                for k in x:
                    ret[k] = torch.cat([ret[k], x[k]], dim=0)
        return ret

    def render_rays(self, pts, bins, model, rays_d):
        N = bins.shape[1]
        dir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        dir = dir[:, None].expand(pts.shape)
        pts_flat = pts.reshape(-1, 3)
        dir_flat = dir.reshape(-1, 3)
        xyz = self.xyz_encoder(pts_flat)
        dir = self.dir_encoder(dir_flat)

        raw = model({'xyz': xyz, 'dir': dir})
        # raw = self.batchify(model, {'xyz': xyz, 'dir': dir})
        rgb = raw['rgb'].reshape(-1, N, 3)
        sig = raw['sig'].reshape(-1, N)
        d = bins[..., 1:] - bins[..., :-1]
        d = torch.cat([d, torch.ones_like(d[..., :1], device=rays_d.device) * 1e10], dim=-1)
        d = d * torch.norm(rays_d[..., None, :], dim=-1)
        t = torch.cumsum((sig * d).squeeze(), dim=-1)
        t = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)[..., :-1]
        t = torch.exp(-t)
        w = t * (1. - torch.exp(-sig * d))
        rgb = torch.sum(w[..., None] * rgb, dim=-2)
        acc = torch.sum(w, -1)

        if cfg.task_arg.white_bkgd:
            rgb = rgb + (1. - acc[..., None])

        return rgb.squeeze(), w.squeeze()

    def render(self, x):
        N_rays = x['rays_o'].shape[0]
        rays_o = x['rays_o']
        rays_d = x['rays_d']
        bins = torch.linspace(self.near, self.far, self.N_c+1, device=rays_o.device)
        bins = bins.expand(N_rays, -1)
        lower = bins[..., :-1]
        upper = bins[..., 1:]
        d = torch.rand(upper.shape, device=rays_o.device)
        t = lower + (upper - lower) * d
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t[..., None]
        c_rgb, c_w = self.render_rays(pts, t, self.nerf, rays_d)


        c_w = c_w[..., 1:-1]
        c_w = c_w + 1e-5
        t_mids = 0.5 * (t[..., 1:] + t[..., :-1])
        pdf = c_w / torch.sum(c_w, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        d = torch.rand([N_rays, self.N_f], device=rays_o.device)
        ids = torch.searchsorted(cdf, d, right=True)
        # lower = torch.max(torch.zeros_like(ids), ids - 1)
        # upper = torch.min(torch.ones_like(ids) * (self.N_c - 1), ids)
        lower = (ids - 1).clamp(min=0, max=t_mids.shape[1]-1)
        upper = ids.clamp(min=0, max=t_mids.shape[1]-1)
        lower_bin = torch.gather(t_mids, -1, lower)
        upper_bin = torch.gather(t_mids, -1, upper)
        lower_cdf = torch.gather(cdf, -1, lower)
        upper_cdf = torch.gather(cdf, -1, upper)
        width = upper_cdf - lower_cdf
        width = torch.where(width < 1e-5, torch.ones_like(width), width)
        t_f = lower_bin + (d - lower_cdf) / width * (upper_bin - lower_bin)
        t_f = t_f.detach()
        t = torch.sort(torch.cat([t, t_f], dim=-1), dim=-1)[0]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t[..., None]
        f_rgb, f_w = self.render_rays(pts, t, self.fine_nerf, rays_d)

        return {'c_rgb': c_rgb, 'f_rgb': f_rgb}

    def forward(self, x):
        N_rays = x['N_rays'].item()
        x['rays_o'] = x['rays_o'].squeeze()
        x['rays_d'] = x['rays_d'].squeeze()
        x['rgb'] = x['rgb'].squeeze()
        ret = {}
        for i in range(0, N_rays, self.chunk):
            x_i = {
                'rays_o': x['rays_o'][i:i+self.chunk],
                'rays_d': x['rays_d'][i:i+self.chunk]
            }
            x_i = self.render(x_i)
            if i == 0:
                for k in x_i:
                    ret[k] = x_i[k]
            else:
                for k in x_i:
                    ret[k] = torch.cat([ret[k], x_i[k]], dim=0)
        return ret