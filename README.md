# EmbodiedMAE

Official implementation of [EmbodiedMAE](https://arxiv.org/abs/2505.10105), a masked autoencoder trained on DROID-3D, for embodied manipulation tasks.

## 1. Installation

Download our source code:
```bash
git clone https://github.com/ZibinDong/embodiedmae.git
cd embodiedmae
```

Create a virtual environment with Python >= 3.10 and activate it, e.g. with miniconda:
```bash
conda create -n embodiedmae python=3.10
conda activate embodiedmae
```

Install the package:
```bash
pip install -e .
```
> Note: If you want to use point cloud processing, please use `pip install -e ".[pc]"` to install `pytorch3d`. Installing `pytorch3d` could be tricky, so we recommend using conda to install it. You can refer to the [pytorch3d installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for more details. And please make sure your pytorch version is compatible with `pytorch3d` which only supports `pytorch>=2.1,<=2.4.1`.

## 2. Usage

You can use the `EmbodiedMAEModel` class to create an instance of the model. Here is an example of how to use it:

```python
from embodied_mae import EmbodiedMAEModel

device = "cuda"

model = EmbodiedMAEModel.from_pretrained("ZibinDong/embodiedmae-large")
model = model.to(device).eval()

rgb = ...  # (*b, c, h, w) in [-1, 1] (h=w=224 is preferred)
depth = ... # (*b, 1, h, w) in [0, 2] (meter) (h=w=224 is preferred)
pc = ...  # (*b, n, 3) (meter) (n=8192 is preferred)

# rgb-only forward
emb = model(rgb, None, None).embedding  # (*b, 196, 768)
# rgbd forward
emb = model(rgb, depth, None).embedding  # (*b, 392, 768)
# rgbpc forward
emb = model(rgb, None, pc).embedding  # (*b, 392, 768) only if pytorch3d is installed
# rgbdpc forward
emb = model(rgb, depth, pc).embedding  # (*b, 588, 768) only if pytorch3d is installed
```

You can also run `test.py` to check if the model works correctly. You are expected to see the output like this:
```
Testing EmbodiedMAEModel forward with full inputs...
EmbodiedMAEModel output shape: torch.Size([2, 588, 1024])
Successfully obtained embeddings!✅
Testing EmbodiedMAEModel forward with RGB inputs only...
EmbodiedMAEModel RGB output shape: torch.Size([2, 196, 1024])
Successfully obtained RGB embeddings!✅
Testing EmbodiedMAEModel forward with Depth and PC inputs only...
EmbodiedMAEModel Depth+PC output shape: torch.Size([2, 392, 1024])
Successfully obtained Depth+PC embeddings!✅
```

## 3. Citation
If you find this code useful, please consider citing our work:

```bibtex
@inproceedings{dong2025embodiedmae,
    title={EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation}, 
    author={Zibin Dong and Fei Ni and Yifu Yuan and Yinchuan Li and Jianye Hao},
    year={2025},
    booktitle={arXiv preprint arXiv:2505.10105}
}
```