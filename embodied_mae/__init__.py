from .configuration_embodiedmae import EmbodiedMAEConfig, get_embodied_mae_config  # noqa: F401
from .modeling_embodiedmae import (
    EmbodiedMAEForMaskedImageModeling,
    EmbodiedMAEModel,
)

EmbodiedMAEConfig.register_for_auto_class()
EmbodiedMAEModel.register_for_auto_class("AutoModel")
EmbodiedMAEForMaskedImageModeling.register_for_auto_class(
    "AutoModelForMaskedImageModeling"
)
