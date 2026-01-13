"""
Semantic segmentation models for building damage assessment.
"""

from .fcn import FCN8s
from .deeplabv3plus import DeepLabV3Plus

__all__ = ['FCN8s', 'DeepLabV3Plus']
