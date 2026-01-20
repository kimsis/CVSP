"""
HierLight-YOLO Implementation
Based on the paper: "HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography"

This implementation adds custom modules to YOLOv8 for improved small object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Conv from ultralytics
from ultralytics.nn.modules import Conv, DWConv, C2f


# ============================================================================
# 1. IRDCB Module (Inverted Residual Depthwise Convolution Block)
# ============================================================================

class IRDCB(nn.Module):
    """
    Inverted Residual Depthwise Convolution Block
    
    Pattern per paper:
    - Conv 1x1 (compress)
    - N × DCB blocks (expand/filter/compress with conditional residual)
    - Concat all intermediate features (skip aggregation)
    - Conv 1x1 (expand to c2)
    - Optional residual to input if c1 == c2
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of DCB blocks to stack
        t (int): Expansion factor inside DCB
        shortcut (bool): Use residual connection when c1==c2
    """
    def __init__(self, c1, c2, n=2, t=2, shortcut=True, *args, **kwargs):
        super().__init__()
        
        # Hidden channels after initial compression
        self.c = int(c2 // 2)
        self.n = int(n)
        
        # Initial 1x1 conv to compress channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        
        # N depthwise convolution blocks (ModuleList for skip aggregation)
        self.blocks = nn.ModuleList([DCB(self.c, t) for _ in range(self.n)])
        
        # Final 1x1 conv to expand back after concatenation of (n+1) branches
        self.cv2 = Conv(self.c * (self.n + 1), c2, 1, 1)
        
        # Residual connection when dimensions match
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        feats = [x]
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        x = torch.cat(feats, dim=1)
        x = self.cv2(x)
        return x + residual if self.add else x


class DCB(nn.Module):
    """
    Depthwise Convolution Block (used inside IRDCB)
    
    Implements the expand-filter-compress pattern:
    1. Expand: 1x1 conv increases channels by factor t
    2. Filter: Two 3x3 depthwise convs for spatial filtering
    3. Compress: 1x1 conv reduces back to original channels
    
    Args:
        c (int): Number of channels
        t (int): Expansion factor
    """
    def __init__(self, c, t=2):
        super().__init__()
        c_expanded = int(c * t)
        
        # Expand phase
        self.cv1 = Conv(c, c_expanded, 1, 1)
        
        # Filter phase - two depthwise convolutions
        self.dw1 = DWConv(c_expanded, c_expanded, k=3, s=1)
        self.dw2 = DWConv(c_expanded, c_expanded, k=3, s=1)
        
        # Compress phase
        self.cv2 = Conv(c_expanded, c, 1, 1)
        
        # Residual connection
        self.add = True

    def forward(self, x):
        residual = x
        
        # Expand
        x = self.cv1(x)
        
        # Filter - first depthwise conv
        x = self.dw1(x)
        
        # Filter - second depthwise conv
        x = self.dw2(x)
        
        # Compress
        x = self.cv2(x)
        
        return x + residual if self.add else x


# ============================================================================
# 2. LDown Module (Lightweight Downsample)
# ============================================================================

class LDown(nn.Module):
    """
    Lightweight Downsample Module
    
    Efficiently reduces spatial dimensions and channel capacity through:
    1. Depthwise convolution for spatial downsampling
    2. 1x1 convolution for channel compression
    
    Args:
        c1 (int): Input channels (auto-provided by YOLO parser)
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride for downsampling (default: 2)
    """
    def __init__(self, c1, c2, k=3, s=2, *args, **kwargs):
        super().__init__()
        
        # Spatial downsampling with depthwise convolution
        self.dw = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.SiLU(inplace=True)
        
        # Channel compression with 1x1 convolution
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        # Spatial downsampling
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Channel compression
        x = self.pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x
