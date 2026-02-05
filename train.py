import sys
import argparse
from typing import Optional
from pathlib import Path
from types import SimpleNamespace
import yaml
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
_patch_applied = False

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
    if scales and isinstance(scales, dict):
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

from ultralytics import YOLO
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Parse CLI argument
parser = argparse.ArgumentParser(description='Train HierLight-YOLO')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .pt to resume training from the same run')
parser.add_argument('--pretrain-ckpt', type=str, default=None, help='Path to a completed pretrain checkpoint to use as input for stages 2 and 3')
cli_args = parser.parse_args()

# Load training configuration from YAML
CONFIG_PATH = 'configs/train_config.yaml'
with open(CONFIG_PATH, 'r') as f:
    _cfg = yaml.safe_load(f) or {}

# Add resume from CLI (always set, defaults to None if not provided)
_cfg['resume'] = cli_args.resume
_cfg['pretrain_ckpt'] = cli_args.pretrain_ckpt

args = SimpleNamespace(**_cfg)

# Scale configurations
scale_configs = {
    'n': {'depth_multiple': 1.00, 'width_multiple': 0.33},
    's': {'depth_multiple': 1.00, 'width_multiple': 0.67},
    'm': {'depth_multiple': 1.00, 'width_multiple': 1.00}
}

# Load model with scale by modifying YAML config
import tempfile
import os
if args.resume and args.pretrain_ckpt:
    print("Error: --resume and --pretrain-ckpt cannot be used together.")
    sys.exit(1)

# Determine base model name for run directories
if args.resume:
    run_dir = Path(args.resume).parent.parent.name
    if run_dir.endswith('-pretrain'):
        model_base = run_dir.removesuffix('-pretrain')
    elif run_dir.endswith('-finetune'):
        model_base = run_dir.removesuffix('-finetune')
    else:
        model_base = run_dir
elif args.pretrain_ckpt:
    # Extract base name from pretrain checkpoint path
    run_dir = Path(args.pretrain_ckpt).parent.parent.name
    if run_dir.endswith('-pretrain'):
        model_base = run_dir.removesuffix('-pretrain')
    else:
        model_base = run_dir
elif args.weights:
    model_base = Path(args.weights).stem  # e.g., "yolov8n" from "yolov8n.pt"
else:
    model_base = f'hierlight-yolo8-{args.scale}'

pretrain_run = f'{model_base}-pretrain'
frozen_run = f'{model_base}-frozen'
unfrozen_run = f'{model_base}-unfrozen'

# Decide starting stage from checkpoint path (folder suffix)
guessed_stage = None
if args.resume:
    run_dir = Path(args.resume).parent.parent.name
    if run_dir.endswith('-pretrain'):
        guessed_stage = 'pretrain'
    elif run_dir.endswith('-frozen'):
        guessed_stage = 'frozen'
    elif run_dir.endswith('-unfrozen'):
        guessed_stage = 'unfrozen'
elif args.pretrain_ckpt:
    # Using a completed pretrain checkpoint → skip to stage 2
    guessed_stage = 'frozen'

# Use explicit start_stage from config if provided, otherwise use guessed stage
start_stage = args.start_stage if hasattr(args, 'start_stage') and args.start_stage else (guessed_stage or 'pretrain')

print(f"DEBUG: args.resume={args.resume}")
print(f"DEBUG: start_stage={start_stage}")
print(f"DEBUG: guessed_stage={guessed_stage}")

# Apply patch for custom modules if configured
if args.custom_model:
    tasks.parse_model = patched_parse_model

if args.resume:
    # Load from checkpoint and resume same training run
    model = YOLO(args.resume)
    # When resuming, most settings are taken from the checkpoint; pass resume=True selectively per stage
elif args.pretrain_ckpt:
    # Load a completed pretrain checkpoint to use for fine-tuning stages 2 and 3
    model = YOLO(args.pretrain_ckpt)
elif args.weights:
    # Load pretrained checkpoint for fine-tuning on a different dataset
    model = YOLO(args.weights)
else:
    # Build model from custom YAML with selected scale (scale only applies here)
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

def _scaled_lr(base_lr: float, freeze_layers: int | None, *, factor: float, min_scale: float) -> float:
    """Scale LR down as a function of frozen layers.
    Linear heuristic: lr = base_lr * max(min_scale, 1 - factor * freeze)
    Defaults: factor=0.05, min_scale=0.10 for safer reductions.
    """
    if not freeze_layers:
        return base_lr
    scale = max(float(min_scale), 1.0 - float(factor) * int(freeze_layers))
    return base_lr * scale

def build_common_train_kwargs(args, run_name: Optional[str], data_path: str) -> dict:
    return {
        'data': data_path,
        'imgsz': 640,
        'batch': args.batch,
        'device': 0,
        'optimizer': 'Adam',
        'momentum': 0.9,
        'weight_decay': args.weight_decay,
        'warmup_epochs': 3,
        'patience': args.patience,
        'save': True,
        'save_period': args.save_period,
        'project': str(Path('../runs/train').resolve()),
        'name': run_name,
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'amp': True,
        'workers': args.workers,
        'close_mosaic': 10,
    }

def run_train(model, common: dict, *, epochs: int, lr0: float, freeze: Optional[int], resume: bool):
    kwargs = dict(common)
    kwargs.update({'epochs': epochs, 'lr0': lr0, 'resume': resume})
    if freeze is not None:
        kwargs['freeze'] = freeze
    # When resuming, remove 'pretrained' to avoid conflicts with resume mode
    if resume and 'pretrained' in kwargs:
        del kwargs['pretrained']
    return model.train(**kwargs)

# Common kwargs for each stage
pretrain_common = build_common_train_kwargs(args, pretrain_run, args.pretrain_data)
frozen_common = build_common_train_kwargs(args, frozen_run, args.finetune_data)
unfrozen_common = build_common_train_kwargs(args, unfrozen_run, args.finetune_data)

# Stage execution flags
do_pretrain = start_stage == 'pretrain'
do_frozen = start_stage in ['pretrain', 'frozen']

# Stage 1: pretrain on baseline dataset
stage1_best = str(Path(f'runs/train/{pretrain_run}/weights/best.pt').resolve())

# Determine starting checkpoint based on stage and available weights
if args.pretrain_ckpt:
    # Explicit pretrain checkpoint provided via CLI
    finetune_start_ckpt = str(Path(args.pretrain_ckpt).resolve())
    do_pretrain = False
elif args.weights and start_stage in ['frozen', 'unfrozen']:
    # Starting from frozen/unfrozen with base weights (e.g., yolov8m.pt)
    finetune_start_ckpt = args.weights
elif args.resume:
    # Resuming from a checkpoint
    finetune_start_ckpt = str(Path(args.resume).resolve())
elif start_stage == 'unfrozen':
    # Starting at unfrozen stage, look for frozen checkpoint first
    frozen_best = str(Path(f'runs/train/{frozen_run}/weights/best.pt').resolve())
    if Path(frozen_best).exists():
        finetune_start_ckpt = frozen_best
    elif Path(stage1_best).exists():
        # Fallback to pretrain checkpoint
        finetune_start_ckpt = stage1_best
    else:
        # No checkpoint found - use the base model (already loaded)
        finetune_start_ckpt = None
elif start_stage == 'frozen':
    # Starting at frozen stage, look for pretrain checkpoint
    if Path(stage1_best).exists():
        finetune_start_ckpt = stage1_best
    else:
        # No checkpoint found - use the base model (already loaded)
        finetune_start_ckpt = None
else:
    # Default: pretrain stage
    finetune_start_ckpt = stage1_best

if do_pretrain:
    resume_pretrain = bool(args.resume)
    
    # Check if resuming a completed pretrain run
    if resume_pretrain:
        # Try to read the checkpoint to see if training is complete
        import torch
        try:
            ckpt = torch.load(args.resume, map_location='cpu')
            epoch = ckpt.get('epoch', -1)
            if epoch >= args.pretrain_epochs - 1:  # 0-indexed, so epoch 99 = 100 epochs
                print(f"Stage 1 (pretrain): SKIPPED - checkpoint already trained to {epoch+1}/{args.pretrain_epochs} epochs")
                resume_pretrain = False
                do_pretrain = False
                finetune_start_ckpt = args.resume  # Use the pretrain checkpoint for finetuning
        except Exception as e:
            print(f"Warning: Could not check checkpoint completion: {e}")
    
    if do_pretrain:
        print(
            f"Stage 1 (pretrain): data=../{args.pretrain_data}, epochs={args.pretrain_epochs}, "
            f"lr0={args.lr0:.6f}, resume={resume_pretrain}"
        )
        results = run_train(
            model,
            pretrain_common,
            epochs=args.pretrain_epochs,
            lr0=args.lr0,
            freeze=None,
            resume=resume_pretrain,
        )
        finetune_start_ckpt = stage1_best
elif args.resume and not args.pretrain_ckpt:
    # Skipping pretrain because resuming from a finetune checkpoint
    finetune_start_ckpt = str(Path(args.resume).resolve())

# Stage 2: optional frozen finetune
if do_frozen and args.freeze is not None and args.freeze_epochs > 0:
    frozen_lr = _scaled_lr(
        args.lr0,
        args.freeze,
        factor=args.freeze_lr_factor,
        min_scale=args.freeze_lr_min_scale,
    )
    print(
        f"Stage 2 (frozen finetune): data=../{args.finetune_data}, freeze={args.freeze}, "
        f"epochs={args.freeze_epochs}, lr0={frozen_lr:.6f}, resume=False"
    )
    model = YOLO(finetune_start_ckpt)
    results = run_train(
        model,
        frozen_common,
        epochs=args.freeze_epochs,
        lr0=frozen_lr,
        freeze=args.freeze,
        resume=False,
    )
    finetune_start_ckpt = str(Path(f'runs/train/{frozen_run}/weights/best.pt').resolve())

# Stage 3: unfrozen finetune
# Determine if Stage 3 should resume (only if it's a partially trained finetune checkpoint)
resume_stage3 = False
if start_stage == 'unfrozen' and args.resume:
    # Check if the finetune checkpoint is incomplete
    try:
        ckpt = torch.load(args.resume, map_location='cpu')
        # Check if this is a full checkpoint (with epoch info) or weights-only
        if isinstance(ckpt, dict) and 'epoch' in ckpt:
            epoch = ckpt.get('epoch', -1)
            print(f"Stage 3: Full checkpoint detected at epoch {epoch}/{args.epochs-1} (0-indexed)")
            if epoch < args.epochs - 1:  # 0-indexed, not yet at final epoch
                resume_stage3 = True
                print(f"Stage 3: Resuming from epoch {epoch+1}/{args.epochs}")
            else:
                print(f"Stage 3: Checkpoint complete, starting fresh training")
        else:
            # Weights-only checkpoint - assume it's incomplete and should be resumed
            resume_stage3 = True
            print(f"Stage 3: Weights-only checkpoint detected, resuming training")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        print(f"Proceeding with resume_stage3=True (attempt to resume)")
        resume_stage3 = True

print(
    f"Stage 3 (unfrozen finetune): data=../{args.finetune_data}, epochs={args.epochs}, "
    f"lr0={args.lr0:.6f}, resume={resume_stage3}"
)

# Load model for stage 3
if finetune_start_ckpt:
    model = YOLO(finetune_start_ckpt)
# else: model is already loaded from earlier

results = run_train(
    model,
    unfrozen_common,
    epochs=args.epochs,
    lr0=args.lr0,
    freeze=0,
    resume=resume_stage3,
)

# Evaluate on validation set
metrics = model.val()

print(f"\nTraining completed!")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
