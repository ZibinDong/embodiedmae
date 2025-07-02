from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

from .configuration_embodiedmae import EmbodiedMAEConfig
from .modular_embodiedmae import (
    EmbodiedMAEDecoder,
    EmbodiedMAEDepthEmbeddings,
    EmbodiedMAEPointCloudEmbeddings,
    EmbodiedMAERGBEmbeddings,
    EncoderModelOutput,
    concat_sequence_with_dummy,
    prepare_shuffle_idx,
)


class EmbodiedMAEModel(PreTrainedModel):
    config_class = EmbodiedMAEConfig

    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__(config)
        self.config = config

        self.dirichlet = torch.distributions.Dirichlet(
            torch.full((3,), config.dirichlet_alpha)
        )

        self.rgb_embeddings = EmbodiedMAERGBEmbeddings(config)
        self.depth_embeddings = EmbodiedMAEDepthEmbeddings(config)
        self.pc_embeddings = EmbodiedMAEPointCloudEmbeddings(config)

        self.encoder = Dinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        self.embedding_sz = (
            num_patches,
            num_patches,
            config.num_pc_centers,
        )  # token size for each modality
        self.unmask_sz = config.unmask_sz  # number of unmasked tokens

    def get_input_embeddings(
        self,
        rgb: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        pc: Optional[torch.Tensor],
        add_mask: bool = True,
        unmask_sz: Optional[int] = None,
        forward_pc: bool = True,
        shuffle_idx: Optional[torch.Tensor] = None,
    ):
        # provide at least one modality
        assert any([rgb is not None, depth is not None, pc is not None])

        # embeddings
        rgb_emb = self.rgb_embeddings(rgb)
        depth_emb = self.depth_embeddings(depth)
        pc_emb, pc_centers, pc_knn = self.pc_embeddings(pc)
        if not forward_pc:
            pc = None
            pc_emb = None

        # concat embeddings
        all_emb = concat_sequence_with_dummy(
            [rgb_emb, depth_emb, pc_emb], self.embedding_sz
        )

        # prepare shuffle indices
        shuffle_idx, restore_idx, unmask_sz = prepare_shuffle_idx(
            has_rgb=rgb is not None,
            has_depth=depth is not None,
            has_pc=pc is not None,
            batch_size=all_emb.shape[0],
            unmask_sz=self.unmask_sz if unmask_sz is None else unmask_sz,
            dirichlet=self.dirichlet,
            embedding_sz=self.embedding_sz,
            add_mask=add_mask,
            shuffle_idx=shuffle_idx,
            device=all_emb.device,
        )

        # get unmasked embeddings
        unmasked_emb = torch.gather(
            all_emb, 1, shuffle_idx[:, :unmask_sz, None].repeat(1, 1, all_emb.shape[-1])
        )

        return EncoderModelOutput(
            embedding=unmasked_emb,
            pc_centers=pc_centers,
            pc_knn=pc_knn,
            shuffle_idx=shuffle_idx,
            restore_idx=restore_idx,
            add_mask=add_mask,
            unmask_sz=unmask_sz,
        )

    def get_last_hidden_states(
        self,
        embedding_output: EncoderModelOutput,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        embedding = embedding_output.embedding

        encoder_outputs = self.encoder(
            embedding,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        embedding_output.last_hidden_states = sequence_output
        embedding_output.hidden_states = encoder_outputs.hidden_states
        embedding_output.attentions = encoder_outputs.attentions

        return embedding_output

    def forward(
        self,
        rgb: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        pc: Optional[torch.Tensor],
        add_mask: bool = False,
        unmask_sz: Optional[int] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        forward_pc: bool = True,
    ):
        embedding_output = self.get_input_embeddings(
            rgb, depth, pc, add_mask, unmask_sz, forward_pc
        )
        return self.get_last_hidden_states(
            embedding_output, output_attentions, output_hidden_states
        )


class EmbodiedMAEForMaskedImageModeling(EmbodiedMAEModel):
    def __init__(self, config: EmbodiedMAEConfig):
        super().__init__(config)
        self.decoder = EmbodiedMAEDecoder(config)

    def forward(
        self,
        rgb: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        pc: Optional[torch.Tensor],
        add_mask: bool = True,
        unmask_sz: Optional[int] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        forward_pc: bool = True,
    ):
        encoder_output = super().forward(
            rgb,
            depth,
            pc,
            add_mask,
            unmask_sz,
            output_attentions,
            output_hidden_states,
            forward_pc,
        )
        decoder_input = self.decoder.get_decoder_input(encoder_output)
        return self.decoder(decoder_input)

    @torch.no_grad()
    def visualize(
        self,
        rgb: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        pc: Optional[torch.Tensor],
        mask_rgb: bool = False,
        mask_depth: bool = False,
        mask_pc: bool = False,
        add_mask: bool = True,
        unmask_sz: Optional[int] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        forward_pc: bool = True,
    ):
        _rgb = None if mask_rgb else rgb
        _depth = None if mask_depth else depth
        _pc = None if mask_pc else pc
        encoder_output = super().forward(
            _rgb,
            _depth,
            _pc,
            add_mask,
            unmask_sz,
            output_attentions,
            output_hidden_states,
            forward_pc,
        )
        decoder_input = self.decoder.get_decoder_input(encoder_output)
        return self.decoder.visualize(decoder_input, rgb, depth, pc)


__all__ = [EmbodiedMAEModel, EmbodiedMAEForMaskedImageModeling]
