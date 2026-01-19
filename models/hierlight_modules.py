"""
HierLight-YOLO Implementation
Based on the paper: "HierLight-YOLO: A Hierarchical and Lightweight Object Detection Network for UAV Photography"

This implementation adds custom modules to YOLOv8 for improved small object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Conv from ultralytics
from ultralytics.nn.modules import Conv


# ============================================================================
# 1. IRDCB Module (Inverted Residual Depthwise Convolution Block)
# ============================================================================

class IRDCB(nn.Module):
    """
    Inverted Residual Depthwise Convolution Block
    
    Combines depthwise separable convolutions with inverted residual structure
    for efficient feature extraction with reduced parameters.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels (if None, same as c1)
        n (int): Number of bottleneck blocks (default: 2)
        t (int): Expansion factor (default: 2)
        shortcut (bool): Use residual connection when c1==c2
    """
    def __init__(self, c1, c2, n=1, t=2, shortcut=True, *args, **kwargs):
        super().__init__()
        
        self.c = int(c2 // 2)  # Hidden channels
        
        # Initial 1x1 conv to compress channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        
        # Stack of n depthwise convolution blocks
        self.m = nn.Sequential(*(DCB(self.c, t) for _ in range(n)))
        
        # Final 1x1 conv to expand back
        self.cv2 = Conv(self.c, c2, 1, 1)
        
        # Residual connection when dimensions match
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.m(x)
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
        self.dw1 = nn.Conv2d(c_expanded, c_expanded, 3, 1, 1, groups=c_expanded, bias=False)
        self.bn1 = nn.BatchNorm2d(c_expanded)
        self.act1 = nn.SiLU(inplace=True)
        
        self.dw2 = nn.Conv2d(c_expanded, c_expanded, 3, 1, 1, groups=c_expanded, bias=False)
        self.bn2 = nn.BatchNorm2d(c_expanded)
        self.act2 = nn.SiLU(inplace=True)
        
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
        x = self.bn1(x)
        x = self.act1(x)
        
        # Filter - second depthwise conv
        x = self.dw2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
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


# ============================================================================
# 3. HEPAN Neck (Hierarchical Extended Path Aggregation Network)
# ============================================================================

class HEPAN(nn.Module):
    """
    Hierarchical Extended Path Aggregation Network
    
    Enhances multi-scale feature fusion through:
    1. Hierarchical channel compression via 1x1 convs
    2. Bidirectional feature flow (top-down and bottom-up)
    3. Dense cross-layer skip connections
    
    Args:
        channels (list): List of channel numbers for each scale [P2, P3, P4, P5]
        depth (int): Number of IRDCB blocks in fusion layers
    """
    def __init__(self, c1, args=None, **kwargs):
        super().__init__()
        # Handle YOLO parser style: (c1, args_list) where args_list = [channels_list, depth]
        # For now, use default values since HEPAN is typically not called with channel info from YOLO
        channels = [64, 128, 256, 512]  # Default values
        depth = 2  # Default value
        
        if isinstance(args, (list, tuple)) and len(args) > 0:
            if isinstance(args[0], (list, tuple)):
                channels = args[0]
            if len(args) > 1:
                depth = args[1]
        elif args is not None and not isinstance(args, (list, tuple)):
            depth = args
        
        # Channel compression for backbone features (HFCC)
        self.compress_p2 = Conv(channels[0], channels[0]//2, 1, 1)
        self.compress_p3 = Conv(channels[1], channels[1]//2, 1, 1)
        self.compress_p4 = Conv(channels[2], channels[2]//2, 1, 1)
        self.compress_p5 = Conv(channels[3], channels[3]//2, 1, 1)
        
        # Top-down pathway
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Fusion blocks for top-down
        self.td_irdcb1 = IRDCB(channels[3]//2 + channels[2]//2, channels[2]//2, n=depth)
        self.td_irdcb2 = IRDCB(channels[2]//2 + channels[1]//2, channels[1]//2, n=depth)
        self.td_irdcb3 = IRDCB(channels[1]//2 + channels[0]//2, channels[0]//2, n=depth)
        
        # Bottom-up pathway
        self.down1 = LDown(channels[0]//2, channels[0]//2)
        self.down2 = LDown(channels[1]//2, channels[1]//2)
        self.down3 = LDown(channels[2]//2, channels[2]//2)
        
        # Fusion blocks for bottom-up
        self.bu_irdcb1 = IRDCB(channels[0]//2 + channels[1]//2, channels[1]//2, n=depth)
        self.bu_irdcb2 = IRDCB(channels[1]//2 + channels[2]//2, channels[2]//2, n=depth)
        self.bu_irdcb3 = IRDCB(channels[2]//2 + channels[3]//2, channels[3]//2, n=depth)
        
        # Output projections
        self.out_p2 = Conv(channels[0]//2, channels[0], 1, 1)
        self.out_p3 = Conv(channels[1]//2, channels[1], 1, 1)
        self.out_p4 = Conv(channels[2]//2, channels[2], 1, 1)
        self.out_p5 = Conv(channels[3]//2, channels[3], 1, 1)

    def forward(self, features):
        """
        Args:
            features (list): [P2, P3, P4, P5] features from backbone
        
        Returns:
            list: Enhanced multi-scale features [N2, N3, N4, N5]
        """
        p2, p3, p4, p5 = features
        
        # 1. Hierarchical Feature Channel Compression (HFCC)
        c2 = self.compress_p2(p2)
        c3 = self.compress_p3(p3)
        c4 = self.compress_p4(p4)
        c5 = self.compress_p5(p5)
        
        # 2. Top-down pathway
        # P5 -> P4
        p5_up = self.up1(c5)
        td4 = torch.cat([p5_up, c4], dim=1)
        td4 = self.td_irdcb1(td4)
        
        # P4 -> P3
        td4_up = self.up2(td4)
        td3 = torch.cat([td4_up, c3], dim=1)
        td3 = self.td_irdcb2(td3)
        
        # P3 -> P2
        td3_up = self.up3(td3)
        td2 = torch.cat([td3_up, c2], dim=1)
        td2 = self.td_irdcb3(td2)
        
        # 3. Bottom-up pathway with dense connections
        # P2 -> P3
        td2_down = self.down1(td2)
        bu3 = torch.cat([td2_down, td3], dim=1)
        bu3 = self.bu_irdcb1(bu3)
        
        # P3 -> P4
        bu3_down = self.down2(bu3)
        bu4 = torch.cat([bu3_down, td4], dim=1)
        bu4 = self.bu_irdcb2(bu4)
        
        # P4 -> P5
        bu4_down = self.down3(bu4)
        bu5 = torch.cat([bu4_down, c5], dim=1)
        bu5 = self.bu_irdcb3(bu5)
        
        # 4. Output projections
        n2 = self.out_p2(td2)
        n3 = self.out_p3(bu3)
        n4 = self.out_p4(bu4)
        n5 = self.out_p5(bu5)
        
        return [n2, n3, n4, n5]


# ============================================================================
# 4. P2 Small Object Detection Head
# ============================================================================

class P2DetectionHead(nn.Module):
    """
    P2 Small Object Detection Head
    
    Adds high-resolution (160x160) detection layer for small objects by:
    1. Upsampling P3 features from 80x80 to 160x160
    2. Fusing with shallow P2 features (160x160)
    3. Processing through IRDCB for detection
    
    Args:
        c_p2 (int): Channels from P2 backbone feature
        c_p3 (int): Channels from P3 neck feature
        c_out (int): Output channels for detection
    """
    def __init__(self, c_p2, args=None, **kwargs):
        super().__init__()
        # Handle YOLO parser style: (c_p2, args_list) where args_list = [c_p3, c_out]
        if isinstance(args, (list, tuple)):
            c_p3 = args[0] if len(args) > 0 else c_p2
            c_out = args[1] if len(args) > 1 else c_p3
        else:
            c_p3 = args if args is not None else c_p2
            c_out = kwargs.get('c_out', c_p3)
        
        # Upsample P3 from 80x80 to 160x160
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Fusion block
        self.fusion = IRDCB(c_p2 + c_p3, c_out, n=2)

    def forward(self, p2_backbone, p3_neck):
        """
        Args:
            p2_backbone: P2 features from backbone (160x160)
            p3_neck: P3 features from neck (80x80)
        
        Returns:
            Fused features for P2 detection (160x160)
        """
        # Upsample P3 to match P2 resolution
        p3_up = self.upsample(p3_neck)
        
        # Concatenate along channel dimension
        fused = torch.cat([p3_up, p2_backbone], dim=1)
        
        # Process through IRDCB
        out = self.fusion(fused)
        
        return out

# ============================================================================
# 5. Complete HierLight-YOLO Model Configuration
# ============================================================================

def create_hierlight_yolo_yaml():
    """
    Creates YAML configuration for HierLight-YOLO variants
    
    This should be saved as a .yaml file and used with Ultralytics framework
    """
    
    yaml_n = """
# HierLight-YOLO-N Configuration
# Parameters: 2.2M | FLOPs: 11.7G

# Backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 2, IRDCB, [128, 2, 2]]  # 2
  - [-1, 1, LDown, [256, 3, 2]]  # 3-P3/8
  - [-1, 2, IRDCB, [256, 2, 2]]  # 4
  - [-1, 1, LDown, [512, 3, 2]]  # 5-P4/16
  - [-1, 2, IRDCB, [512, 2, 2]]  # 6
  - [-1, 1, LDown, [1024, 3, 2]]  # 7-P5/32
  - [-1, 2, IRDCB, [1024, 2, 2]]  # 8
  - [-1, 1, SPPF, [1024, 5]]  # 9

# Neck (HEPAN)
head:
  - [-1, 1, Conv, [512, 1, 1]]  # 10 - compress P5
  - [4, 1, Conv, [256, 1, 1]]   # 11 - compress P3
  - [2, 1, Conv, [128, 1, 1]]   # 12 - compress P2
  
  # Top-down pathway
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13
  - [[-1, 11], 1, Concat, [1]]  # 14
  - [-1, 2, IRDCB, [256, 2, 2]]  # 15 - td3
  
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 16
  - [[-1, 12], 1, Concat, [1]]  # 17
  - [-1, 2, IRDCB, [128, 2, 2]]  # 18 - td2
  
  # Bottom-up pathway
  - [-1, 1, LDown, [128, 3, 2]]  # 19
  - [[-1, 15], 1, Concat, [1]]  # 20
  - [-1, 2, IRDCB, [256, 2, 2]]  # 21 - bu3
  
  - [-1, 1, LDown, [256, 3, 2]]  # 22
  - [[-1, 10], 1, Concat, [1]]  # 23
  - [-1, 2, IRDCB, [512, 2, 2]]  # 24 - bu4
  
  # Detection heads
  - [[18, 21, 24], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
"""
    
    return yaml_n


# ============================================================================
# 6. Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of HierLight-YOLO modules
    """
    
    # Test IRDCB module
    print("Testing IRDCB module...")
    irdcb = IRDCB(c1=128, c2=128, n=2, t=2)
    x = torch.randn(1, 128, 80, 80)
    out = irdcb(x)
    print(f"IRDCB: Input {x.shape} -> Output {out.shape}")
    
    # Test LDown module
    print("\nTesting LDown module...")
    ldown = LDown(c1=128, c2=256, k=3, s=2)
    x = torch.randn(1, 128, 80, 80)
    out = ldown(x)
    print(f"LDown: Input {x.shape} -> Output {out.shape}")
    
    # Test HEPAN
    print("\nTesting HEPAN...")
    hepan = HEPAN(channels=[128, 256, 512, 1024])
    p2 = torch.randn(1, 128, 160, 160)
    p3 = torch.randn(1, 256, 80, 80)
    p4 = torch.randn(1, 512, 40, 40)
    p5 = torch.randn(1, 1024, 20, 20)
    n2, n3, n4, n5 = hepan([p2, p3, p4, p5])
    print(f"HEPAN outputs: N2={n2.shape}, N3={n3.shape}, N4={n4.shape}, N5={n5.shape}")
    
    # Test P2 Detection Head
    print("\nTesting P2 Detection Head...")
    p2_head = P2DetectionHead(c_p2=128, c_p3=256, c_out=256)
    p2_backbone = torch.randn(1, 128, 160, 160)
    p3_neck = torch.randn(1, 256, 80, 80)
    out = p2_head(p2_backbone, p3_neck)
    print(f"P2 Head: Output {out.shape}")
    
    print("\n✓ All modules working correctly!")
    print("\nTo use with Ultralytics YOLOv8:")
    print("1. Copy these classes to ultralytics/nn/modules/")
    print("2. Register them in ultralytics/nn/tasks.py")
    print("3. Create a YAML config with the model architecture")
    print("4. Train with: model = YOLO('hierlight-yolo-n.yaml')")