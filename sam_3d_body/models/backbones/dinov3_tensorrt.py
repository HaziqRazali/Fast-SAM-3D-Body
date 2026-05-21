# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
TensorRT wrapper for DINOv3 Backbone.

Usage:
    from sam_3d_body.models.backbones.dinov3_tensorrt import TRTDinov3Backbone

    backbone = TRTDinov3Backbone(engine_path="path/to/backbone.engine")
    output = backbone(input_tensor)
"""

import os
import torch
from torch import nn


class TRTDinov3Backbone(nn.Module):
    """
    TensorRT inference wrapper for DINOv3 backbone.

    This class provides a drop-in replacement for the PyTorch DINOv3 backbone
    using a pre-built TensorRT engine for faster inference.
    """

    def __init__(
        self,
        engine_path: str,
        embed_dim: int = 1280,
        patch_size: int = 16,
        output_size: tuple = (32, 32),
    ):
        """
        Args:
            engine_path: Path to TensorRT engine file
            embed_dim: Feature embedding dimension (default: 1280 for dinov3_vith16plus)
            patch_size: Patch size of the ViT model
            output_size: Spatial size of output feature map (H', W')
        """
        super().__init__()

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.engine_path = engine_path
        self.embed_dim = embed_dim
        self.embed_dims = embed_dim  # Alias for compatibility
        self.patch_size = patch_size
        self.output_size = output_size

        # Lazy initialization - engine loaded on first forward
        self._engine = None
        self._context = None
        self._initialized = False

        # Pre-allocated I/O buffers (created on first forward, reused every frame).
        # Stable GPU addresses let the surrounding torch.compile CUDA graph skip
        # the input-copy step it would otherwise need for dynamic-address tensors.
        self._input_buffer: torch.Tensor | None = None
        self._output_buffer: torch.Tensor | None = None

        # Actual spatial size of the TRT engine's input/output (read from engine at
        # init time).  May differ from output_size * patch_size when the engine was
        # built for a smaller resolution than the rest of the model expects.
        self._trt_h: int | None = None
        self._trt_w: int | None = None

    def _init_engine(self):
        """Initialize TensorRT engine (lazy loading)."""
        if self._initialized:
            return

        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is required for TRTDinov3Backbone. "
                "Install with: pip install tensorrt"
            )

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"[TRTDinov3Backbone] Loading engine from: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self._engine = self.runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Get binding names
        self.input_name = "input"
        self.output_name = "output"

        # Read the engine's actual fixed spatial input dimensions.
        # For a dynamic-batch engine the shape is (-1, 3, H, W); spatial dims are fixed.
        trt_in_shape = self._engine.get_tensor_shape(self.input_name)
        self._trt_h = int(trt_in_shape[2])
        self._trt_w = int(trt_in_shape[3])

        self._initialized = True
        print(f"[TRTDinov3Backbone] Engine loaded successfully (TRT input: {self._trt_h}×{self._trt_w})")

    def forward(self, x, extra_embed=None):
        """
        Run TensorRT inference.

        Args:
            x: [B, 3, H, W] input tensor (any dtype, will be converted to FP16)
            extra_embed: Not supported in TensorRT mode (ignored)

        Returns:
            output: [B, C, H', W'] feature map (FP16, same as PyTorch backbone output)
        """
        if extra_embed is not None:
            raise NotImplementedError("extra_embed is not supported in TensorRT mode")

        # Lazy init
        if not self._initialized:
            self._init_engine()

        batch_size = x.shape[0]

        # Resize input to the TRT engine's fixed spatial dimensions if needed.
        # This allows the engine to run at a smaller resolution than IMAGE_SIZE
        # (e.g. engine built for 384×384 while the rest of the model uses 512×512).
        if x.shape[2] != self._trt_h or x.shape[3] != self._trt_w:
            x_in = torch.nn.functional.interpolate(
                x.float(), size=(self._trt_h, self._trt_w),
                mode='bilinear', align_corners=False
            ).half()
        else:
            x_in = x.half() if x.dtype != torch.float16 else x

        # Allocate persistent I/O buffers sized for the TRT engine (on first call
        # or if batch size changes).  Buffer sizes match engine dims, not IMAGE_SIZE.
        if self._input_buffer is None or self._input_buffer.shape[0] != batch_size:
            self._input_buffer = torch.empty(
                batch_size, 3, self._trt_h, self._trt_w,
                device=x.device, dtype=torch.float16
            )
            trt_out_h = self._trt_h // self.patch_size
            trt_out_w = self._trt_w // self.patch_size
            self._output_buffer = torch.empty(
                batch_size, self.embed_dim, trt_out_h, trt_out_w,
                device=x.device, dtype=torch.float16
            )
            self._context.set_input_shape(self.input_name, tuple(self._input_buffer.shape))
            self._context.set_tensor_address(self.input_name, self._input_buffer.data_ptr())
            self._context.set_tensor_address(self.output_name, self._output_buffer.data_ptr())

        # Copy (possibly resized) input into the static TRT buffer
        self._input_buffer.copy_(x_in)

        # Execute asynchronously (PyTorch will synchronize when needed)
        self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        # Bilinear-upsample output to the expected spatial size (output_size) if the
        # TRT engine output is smaller (e.g. 24×24 from 384-engine vs 32×32 expected).
        trt_out_h = self._trt_h // self.patch_size
        trt_out_w = self._trt_w // self.patch_size
        exp_h, exp_w = self.output_size
        if trt_out_h != exp_h or trt_out_w != exp_w:
            return torch.nn.functional.interpolate(
                self._output_buffer.float(), size=(exp_h, exp_w),
                mode='bilinear', align_corners=False
            ).half()

        return self._output_buffer

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """
        Get layer depth - returns a dummy value since TensorRT is opaque.

        This method exists for API compatibility with the PyTorch backbone.
        """
        # Return a reasonable default
        return 0, 1


def create_tensorrt_backbone(
    engine_path: str = None,
    checkpoint_dir: str = None,
    name: str = "dinov3_vith16plus",
    **kwargs
):
    """
    Factory function to create TensorRT backbone.

    Args:
        engine_path: Direct path to .engine file
        checkpoint_dir: Directory containing backbone_trt/backbone_dinov3_fp16.engine
        name: Backbone name (used to determine embed_dim and patch_size)

    Returns:
        TRTDinov3Backbone instance
    """
    # Determine engine path
    if engine_path is None:
        if checkpoint_dir is None:
            checkpoint_dir = "./checkpoints/sam-3d-body-dinov3"
        engine_path = os.path.join(checkpoint_dir, "backbone_trt", "backbone_dinov3_fp16.engine")

    # Determine model params based on name
    model_configs = {
        "dinov3_vith16plus": {"embed_dim": 1280, "patch_size": 16},
        "dinov3_vith16": {"embed_dim": 1280, "patch_size": 16},
        "dinov3_vitl16": {"embed_dim": 1024, "patch_size": 16},
        "dinov3_vitb16": {"embed_dim": 768, "patch_size": 16},
        "dinov3_vits16": {"embed_dim": 384, "patch_size": 16},
        "dinov3_vits16plus": {"embed_dim": 384, "patch_size": 16},
        "dinov3_vit7b": {"embed_dim": 1536, "patch_size": 16},
    }

    config = model_configs.get(name, {"embed_dim": 1280, "patch_size": 16})

    # Calculate output size based on default 512x512 input
    image_size = kwargs.get("image_size", (512, 512))
    output_size = (image_size[0] // config["patch_size"], image_size[1] // config["patch_size"])

    return TRTDinov3Backbone(
        engine_path=engine_path,
        embed_dim=config["embed_dim"],
        patch_size=config["patch_size"],
        output_size=output_size,
    )
