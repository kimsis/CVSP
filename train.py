import sys
import argparse
sys.path.insert(0, "./models")

# Import your custom layers
from hierlight_modules import IRDCB, LDown

# Register them into Ultralytics namespace (both locations)
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

# Add to modules
modules.IRDCB = IRDCB
modules.LDown = LDown

# Add to tasks globals (where parse_model looks)
tasks.IRDCB = IRDCB
tasks.LDown = LDown

# Patch parse_model to include custom modules in base_modules and repeat_modules
_original_parse_model = tasks.parse_model

def patched_parse_model(d, ch, verbose=True):
    """Wrapper to inject custom modules into base_modules and repeat_modules"""
    import torch
    from ultralytics.nn.modules import (
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f,
        C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR,
        C3Ghost, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        Concat, Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB,
        ImagePoolingAttn, v10Detect, AIFI, HGStem, HGBlock, ResNetLayer,
        RTDETRDecoder, CBLinear, CBFuse, TorchVision, Index
    )
    from ultralytics.utils.ops import make_divisible
    from ultralytics.utils import LOGGER
    import ast
    import contextlib
    
    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        scale_config = scales[scale]
        depth = scale_config.get("depth_multiple", depth)
        width = scale_config.get("width_multiple", width)
        max_channels = scale_config.get("max_channels", max_channels)

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"activation: {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    
    # Base modules - standard modules that get channel dimensions injected
    base_modules = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f,
        C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR,
        C3Ghost, DWConvTranspose2d, C3x, RepC3, PSA, SCDown, C2fCIB, A2C2f,
        IRDCB, LDown  # Custom modules
    })
    
    # Repeat modules - modules that support n (repeat count) parameter
    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x,
        RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        IRDCB  # Only IRDCB supports repeats, not LDown
    })
    
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m] if m in globals() else locals()[m]
        )
        
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])
            
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset({Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, ImagePoolingAttn, v10Detect}):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]
        
        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    
    return torch.nn.Sequential(*layers), sorted(save)

# Monkey-patch parse_model
tasks.parse_model = patched_parse_model

from ultralytics import YOLO
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train HierLight-YOLO')
parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm'], help='Model scale (n/s/m)')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pt to resume from')
args = parser.parse_args()

# Scale configurations
scale_configs = {
    'n': {'depth_multiple': 1.00, 'width_multiple': 0.33},
    's': {'depth_multiple': 1.00, 'width_multiple': 0.67},
    'm': {'depth_multiple': 1.00, 'width_multiple': 1.00}
}

# Load model with scale by modifying YAML config
import yaml
import tempfile
import os
if args.resume:
    # Load from checkpoint and resume
    model = YOLO(args.resume)
    # When resuming, most settings are taken from the checkpoint; pass resume=True below
else:
    # Build model from custom YAML with selected scale
    model_path = 'models/hierlight-yolo8.yaml'
    with open(model_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override the depth and width multipliers directly
    cfg['depth_multiple'] = scale_configs[args.scale]['depth_multiple']
    cfg['width_multiple'] = scale_configs[args.scale]['width_multiple']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    model = YOLO(tmp_path, task='detect')

    # Clean up temp file after model is loaded
    os.unlink(tmp_path)

# Train
results = model.train(
    data='datasets/VisDrone-DET/data.yaml',
    epochs=100,
    imgsz=640,
    batch=64,                    # Adjust based on GPU memory
    device=0,                    # GPU device
    optimizer='Adam',
    lr0=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=20,                 # Early stopping
    save=True,
    save_period=25,             # Save checkpoint every 25 epochs
    project='.',
    name=f'hierlight-yolo8-{args.scale}' if not args.resume else None,
    exist_ok=True,
    pretrained=True,
    verbose=True,
    seed=0,
    deterministic=True,
    amp=True,                   # Automatic Mixed Precision
    workers=8,
    close_mosaic=10,            # Disable mosaic augmentation in last 10 epochs
    resume=bool(args.resume),   # Enable resume if checkpoint provided
)

# Evaluate on validation set
metrics = model.val()

print(f"\nTraining completed!")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")