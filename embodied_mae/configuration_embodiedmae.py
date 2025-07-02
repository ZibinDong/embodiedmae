import warnings

from transformers import PretrainedConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config


class EmbodiedMAEConfig(PretrainedConfig):
    model_type = "EmbodiedMAE"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        qkv_bias: bool = True,
        apply_layernorm: bool = True,
        attn_implementation: str = "eager",
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        use_swiglu_ffn: bool = False,
        image_size: int = 224,
        patch_size: int = 16,
        num_pc_centers: int = 196,
        num_pc_knn: int = 64,
        dirichlet_alpha: int = 1.0,
        unmask_sz: int = 98,
        decoder_hidden_size: int = 512,
        decoder_num_hidden_layers: int = 4,
        decoder_num_attention_heads: int = 8,
        norm_pix_loss: int = False,
        enable_point_cloud: bool = True,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        self.image_size = image_size
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.apply_layernorm = apply_layernorm
        self.num_pc_centers = num_pc_centers
        self.num_pc_knn = num_pc_knn
        self.dirichlet_alpha = dirichlet_alpha
        self.unmask_sz = unmask_sz

        self._attn_implementation = attn_implementation
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_swiglu_ffn = use_swiglu_ffn

        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_num_attention_heads = decoder_num_attention_heads

        self.norm_pix_loss = norm_pix_loss

        try:
            import pytorch3d  # noqa: F401
        except ImportError:
            enable_point_cloud = False
            warnings.warn(
                "pytorch3d is not installed, disabling point cloud processing."
            )
        self.enable_point_cloud = enable_point_cloud

        super().__init__(**kwargs)


BACKBONE_KWARGS = {
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "mlp_ratio",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "initializer_range",
    "qkv_bias",
    "apply_layernorm",
    "attn_implementation",
    "layerscale_value",
    "drop_path_rate",
    "layer_norm_eps",
    "hidden_act",
    "use_swiglu_ffn",
}


# get config for different size
def get_embodied_mae_config(size: str = "base") -> EmbodiedMAEConfig:
    backbone_config = Dinov2Config.from_pretrained(f"facebook/dinov2-{size}")
    kwargs = {
        k: v for k, v in backbone_config.to_dict().items() if k in BACKBONE_KWARGS
    }
    norm_pix_loss = True if size == "giant" else False
    return EmbodiedMAEConfig(**kwargs, norm_pix_loss=norm_pix_loss)


__all__ = [EmbodiedMAEConfig, get_embodied_mae_config]
