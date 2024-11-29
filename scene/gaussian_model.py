import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement

from errno import EEXIST


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        os.makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise


C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
def SH2RGB(sh):
    return sh * C0 + 0.5

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianModelLite:
    def __init__(self):
        self.max_sh_degree = 3
        self.active_sh_degree = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
        )

    def restore(self, model_args):
        (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
        ) = model_args

    def _colors_precomp_to_features(self, colors_precomp):
        colors = torch.clamp_min(colors_precomp, 0.0).cpu().numpy()  # ?
        # colors = colors_precomp.cpu().numpy()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        return features

    def restore_from_scene_data(self, scene_data: dict) -> None:
        """
        :param scene_data: a dict with keys {'means3D', 'colors_precomp', 'rotations', 'opacities', 'scales', 'means2D'}
        """
        # scene_data = dict((k, v[:20_000]) for k, v in scene_data.items())
        features = self._colors_precomp_to_features(scene_data['colors_precomp'])

        self._xyz = scene_data['means3D']
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = self.scaling_inverse_activation(scene_data['scales'])
        self._rotation = scene_data['rotations']
        self._opacity = self.opacity_inverse_activation(scene_data['opacities'])

    def restore_from_params_data(self, params_data) -> None:
        features = self._colors_precomp_to_features(params_data['colors_precomp'])

        self._xyz = params_data['means3D']
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = params_data['scales']
        self._rotation = params_data['rotations']
        self._opacity = params_data['opacities']

    @staticmethod
    def create_from_params(params: dict):
        gaussians = GaussianModelLite()
        features = gaussians._colors_precomp_to_features(params['rgb_colors'])
        gaussians._xyz = params['means3D']
        gaussians._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        gaussians._scaling = params['log_scales']
        gaussians._rotation = params['unnorm_rotations']
        gaussians._opacity = params['logit_opacities']
        return gaussians

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



