from .base_roi_head import BaseRoIHead
from .bbox_heads import (BBoxHead, ConvFCBBoxHead, Shared2FCBBoxHead)
from .roi_extractors import SingleRoIExtractor
from .standard_roi_head import StandardRoIHead
from .standard_roi_head_oamil import StandardRoIHeadOAMIL

__all__ = [
    'BaseRoIHead', 'BBoxHead',
    'ConvFCBBoxHead', 'Shared2FCBBoxHead', 'StandardRoIHead',
    'SingleRoIExtractor', 'StandardRoIHeadOAMIL',
]
