import warnings

import torch
from termcolor import cprint

from embodied_mae import EmbodiedMAEForMaskedImageModeling, EmbodiedMAEModel

device = "cuda"

# dummy inputs
rgb = (
    torch.rand((2, 3, 224, 224), device=device) * 2.0 - 1.0
)  # (*b, c, h, w) in [-1, 1]
depth = (
    torch.rand((2, 1, 224, 224), device=device) * 2.0
)  # (*b, 1, h, w) in [ 0, 2] (meter)

try:
    import pytorch3d  # noqa: F401

    pc = torch.rand((2, 8192, 3), device=device)  # (*b, 8192, 3) (meter)
except ImportError:
    warnings.warn("pytorch3d is not installed, disabling point cloud processing.")
    pc = None

size = "large"

model = EmbodiedMAEModel.from_pretrained(f"ZibinDong/embodiedmae-{size}")
model = model.to(device).eval()

generation_model = EmbodiedMAEForMaskedImageModeling.from_pretrained(
    f"ZibinDong/embodiedmae-{size}"
)
generation_model = generation_model.to(device).eval()


with torch.no_grad():
    cprint("Testing EmbodiedMAEModel forward with full inputs...", "yellow")
    emb = model(rgb, depth, pc, add_mask=False).embedding
    print(f"EmbodiedMAEModel output shape: {emb.shape}")
    cprint("Successfully obtained embeddings!✅", "green")

    cprint("Testing EmbodiedMAEModel forward with RGB inputs only...", "yellow")
    emb_rgb = model(rgb, None, None, add_mask=False).embedding
    print(f"EmbodiedMAEModel RGB output shape: {emb_rgb.shape}")
    cprint("Successfully obtained RGB embeddings!✅", "green")

    cprint(
        "Testing EmbodiedMAEModel forward with Depth and PC inputs only...", "yellow"
    )
    emb_depth_pc = model(None, depth, pc, add_mask=False).embedding
    print(f"EmbodiedMAEModel Depth+PC output shape: {emb_depth_pc.shape}")
    cprint("Successfully obtained Depth+PC embeddings!✅", "green")
