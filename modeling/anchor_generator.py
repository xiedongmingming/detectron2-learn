# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import math

from typing import List

import torch

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, move_device_like
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry

ANCHOR_GENERATOR_REGISTRY = Registry("ANCHOR_GENERATOR")
ANCHOR_GENERATOR_REGISTRY.__doc__ = """
Registry for modules that creates object detection anchors for feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
"""


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        
        super().__init__()
        
        for i, buffer in enumerate(buffers):
            #
            # Use non-persistent buffer so the values are not saved in checkpoint
            #
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        
        return len(self._buffers)

    def __iter__(self):
        
        return iter(self._buffers.values())


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor
):
    #
    # size：torch.Size([256, 184]) stride：4 base_anchors: {Tensor：(3, 4)}
    #
    grid_height, grid_width = size # 256, 184
    
    shifts_x = move_device_like(
        torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32), target_device_tensor,
    )
    
    shifts_y = move_device_like(
        torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32), target_device_tensor,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(
        params, collections.abc.Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."

    assert len(params), f"{name} in anchor generator cannot be empty!"

    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]

        return [params] * num_features

    if len(params) == 1:

        return list(params) * num_features

    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    
    return params


@ANCHOR_GENERATOR_REGISTRY.register()
class DefaultAnchorGenerator(nn.Module):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides  # [4, 8, 16, 32, 64]

        self.num_features = len(self.strides)  # 5
        
        sizes = _broadcast_params(sizes, self.num_features, "sizes")  # [[32], [64], [128], [256], [512]]
        
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")  # [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios) # BufferList() {list:5(Tensor:{3,4})}

        self.offset = offset
        
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,  # [[32, 64, 128, 256, 512]]
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,  # [[0.5, 1.0, 2.0]]
            "strides": [x.stride for x in input_shape],  # [4, 8, 16, 32, 64]
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,  # 0.0
        }

    def _calculate_anchors(self, sizes, aspect_ratios):
        
        cell_anchors = [  # {list:5} {Tensor:(3,4)} -- 5个级别 3个比例 4个坐标
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)  # s,a:[32][0.5,1.0.2.0]
        ]
        
        return BufferList(cell_anchors)

    @property
    @torch.jit.unused
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    @torch.jit.unused
    def num_anchors(self):  # [3, 3, 3, 3, 3]
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):  # [torch.Size([256, 184]), torch.Size([128, 92]), torch.Size([64, 46]), torch.Size([32, 23]), torch.Size([16, 12])]
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
        """
        anchors = []
        
        # buffers() not supported by torchscript. use named_buffers() instead
        
        buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
        
        for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
            #
            # size：torch.Size([256, 184]) stride：4 base_anchors: {Tensor：(3, 4)}
            #
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors) # {Tensor：(47104,)} {Tensor：(47104,)}
            
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # {Tensor：(47104, 4)}

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)): # s,a:[32][0.5,1.0.2.0]
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        
        for size in sizes:
            
            area = size**2.0
            
            for aspect_ratio in aspect_ratios:
                #
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                #
                w = math.sqrt(area / aspect_ratio)
                
                h = aspect_ratio * w
                
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                
                anchors.append([x0, y0, x1, y1])
                
        return torch.tensor(anchors)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[Boxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        # [
        #     {torch.Size(256, 184)},
        #     {torch.Size(128, 92)},
        #     {torch.Size(64, 46)},
        #     {torch.Size(32, 23)},
        #     {torch.Size(16, 12)}
        # ]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)  # pyre-ignore
        # [
        #     {Tensor：(141312, 4)}, 256x184x3
        #     {Tensor：(35328, 4)},  128x92x3
        #     {Tensor：(8832, 4)},   64x46x3
        #     {Tensor：(2208, 4)},   32x23x3
        #     {Tensor：(576, 4)},    16x12x3
        # ]
        return [Boxes(x) for x in anchors_over_all_feature_maps]


@ANCHOR_GENERATOR_REGISTRY.register()
class RotatedAnchorGenerator(nn.Module):
    """
    Compute rotated anchors used by Rotated RPN (RRPN), described in
    "Arbitrary-Oriented Scene Text Detection via Rotation Proposals".
    """

    box_dim: int = 5
    """
    the dimension of each anchor box.
    """

    @configurable
    def __init__(self, *, sizes, aspect_ratios, strides, angles, offset=0.5):
        """
        This interface is experimental.

        Args:
            sizes (list[list[float]] or list[float]):
                If sizes is list[list[float]], sizes[i] is the list of anchor sizes
                (i.e. sqrt of anchor area) to use for the i-th feature map.
                If sizes is list[float], the sizes are used for all feature maps.
                Anchor sizes are given in absolute lengths in units of
                the input image; they do not dynamically scale if the input image size changes.
            aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
                (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
            strides (list[int]): stride of each input feature.
            angles (list[list[float]] or list[float]): list of angles (in degrees CCW)
                to use for anchors. Same "broadcast" rule for `sizes` applies.
            offset (float): Relative offset between the center of the first anchor and the top-left
                corner of the image. Value has to be in [0, 1).
                Recommend to use 0.5, which means half stride.
        """
        super().__init__()

        self.strides = strides
        
        self.num_features = len(self.strides)
        
        sizes = _broadcast_params(sizes, self.num_features, "sizes")
        
        aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
        
        angles = _broadcast_params(angles, self.num_features, "angles")
        
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)

        self.offset = offset
        
        assert 0.0 <= self.offset < 1.0, self.offset

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        
        return {
            "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
            "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
            "strides": [x.stride for x in input_shape],
            "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
            "angles": cfg.MODEL.ANCHOR_GENERATOR.ANGLES,
        }

    def _calculate_anchors(self, sizes, aspect_ratios, angles):
        
        cell_anchors = [
            self.generate_cell_anchors(size, aspect_ratio, angle).float()
            for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)
        ]
        
        return BufferList(cell_anchors)

    @property
    def num_cell_anchors(self):
        """
        Alias of `num_anchors`.
        """
        return self.num_anchors

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios, 2 sizes and 5 angles, the number of anchors is 30.
                (See also ANCHOR_GENERATOR.SIZES, ANCHOR_GENERATOR.ASPECT_RATIOS
                and ANCHOR_GENERATOR.ANGLES in config)

                In standard RRPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _grid_anchors(self, grid_sizes: List[List[int]]):
        
        anchors = []
        
        for size, stride, base_anchors in zip(
            grid_sizes,
            self.strides,
            self.cell_anchors._buffers.values(),
        ):
            
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
            
            zeros = torch.zeros_like(shift_x)
            
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)

            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))

        return anchors

    def generate_cell_anchors(
        self,
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.5, 1, 2),
        angles=(-90, -60, -30, 0, 30, 60, 90),
    ):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes, aspect_ratios, angles centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):
            angles (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
                storing anchor boxes in (x_ctr, y_ctr, w, h, angle) format.
        """
        anchors = []
        
        for size in sizes:
            
            area = size**2.0
            
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                
                h = aspect_ratio * w
                
                anchors.extend([0, 0, w, h, a] for a in angles)

        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[RotatedBoxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        
        return [RotatedBoxes(x) for x in anchors_over_all_feature_maps]


def build_anchor_generator(cfg, input_shape):
    """
    Built an anchor generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    anchor_generator = cfg.MODEL.ANCHOR_GENERATOR.NAME  # DefaultAnchorGenerator

    return ANCHOR_GENERATOR_REGISTRY.get(anchor_generator)(cfg, input_shape)
