"""The conditionally independent event stream GPT model."""
from typing import Any

import torch

from ..data.types import DataModality, PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig
from .generation.generation_utils import StructuredGenerationMixin
from .model_output import (
    GenerativeOutputLayerBase,
    GenerativeSequenceModelLabels,
    GenerativeSequenceModelLosses,
    GenerativeSequenceModelOutput,
    GenerativeSequenceModelPredictions,
)
from ..data.data_embedding_layer import DataEmbeddingLayer
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)

class EveryQueryOutputLayer(torch.nn.Module):

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()
        self.proj = torch.nn.Linear(config.hidden_size * 2, 1) 
        # self.proj = torch.nn.Linear(config.hidden_size*2 + config.query_hidden_size, 1) 
        # (todo) update config to include query_hidden_size 

    def forward(
        self,
        encoded_context: torch.FloatTensor,
        encoded_query: torch.FloatTensor,
        answer: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        assert encoded_context.shape == encoded_query.shape, f"encoded_context {encoded_context.shape} and encoded_query {encoded_query.shape} should be (batch_size, hidden_size)"
        
        embed = self.proj(torch.cat([encoded_context, encoded_query], dim=1)) 
        
        # torch.nn.functional.elu has Image (-1, 1), but we need our rate parameter to be > 0. So we need to
        # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `embed`.
        rate = torch.nn.functional.elu(embed) + 1 + torch.finfo(embed.dtype).tiny
        rate = rate.squeeze(dim=-1) # Squeeze from (batch_size, 1) to (batch_size)
        
        loss = - torch.distributions.Poisson(rate).log_prob(answer).mean()
        manual_loss = torch.mean( rate - (answer * torch.log(rate)) )
        dloss_drate = torch.mean( 1 - ( answer / rate ) )
        
        out = {
            'loss':loss, 
            'manual_loss':manual_loss, 
            'predicted_rate':rate.squeeze(),
            'unnormalized_rate':embed.squeeze(),
            'dloss_drate': dloss_drate.squeeze(),
        }
        return out 
        

class CIPPTForGenerativeSequenceModeling(StructuredTransformerPreTrainedModel): 
    """The end-to-end model for conditionally independent generative sequence modelling.

    This model is a subclass of :class:`~transformers.StructuredTransformerPreTrainedModel` and is designed
    for generative pre-training over "event-stream" data, with inputs in the form of `PytorchBatch` objects.
    It is trained to solve the generative, multivariate, masked temporal point process problem over the
    defined measurements in the input data.

    This model largely simply passes the input data through a
    `ConditionallyIndependentPointProcessTransformer` followed by a
    `EveryQueryOutputLayer`.

    Args:
        config: The overall model configuration.

    Raises:
        ValueError: If the model configuration does not indicate conditionally independent mode.
    """

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.context_encoder = ConditionallyIndependentPointProcessTransformer(config)
        self.separate_query_embedding_layer = DataEmbeddingLayer(
            n_total_embeddings=config.vocab_size,
            out_dim=config.hidden_size,
            categorical_embedding_dim=config.categorical_embedding_dim,
            numerical_embedding_dim=config.numerical_embedding_dim,
            static_embedding_mode=config.static_embedding_mode,
            split_by_measurement_indices=None,
            do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
            static_weight=config.static_embedding_weight,
            dynamic_weight=config.dynamic_embedding_weight,
            categorical_weight=config.categorical_embedding_weight,
            numerical_weight=config.numerical_embedding_weight,
        )
        self.query_embedding_layer = self.separate_query_embedding_layer.query_embedding
        # uncomment l626 in generative_modelling.py if using separate embedding layers 
        # self.query_embedding_layer = self.context_encoder.input_layer.data_embedding_layer.query_embedding
        self.query_encoder = None # MLP ?? 
        self.output_layer = EveryQueryOutputLayer(config)
    
    def safe_max_seq_dim(self, X: torch.Tensor, mask: torch.BoolTensor):
        # X is batch_size, seq_len, hidden_dim
        mask = mask.unsqueeze(-1).expand_as(X) 
        masked_X = torch.where(mask, X, -float("inf"))
        maxes = masked_X.max(1)[0]
        return torch.nan_to_num(maxes, nan=None, posinf=None, neginf=0)

    def forward(self, batch: PytorchBatch, **kwargs) -> torch.FloatTensor: 
        context, query, answer = batch
        encoded_context = self.context_encoder(context, **kwargs)
        encoded_context = self.safe_max_seq_dim(encoded_context.last_hidden_state, context.event_mask)
        query_embed = self.query_embedding_layer(query, **kwargs) 
        encoded_query = query_embed # (todo) self.query_encoder(query_embed, **kwargs)
        output = self.output_layer(encoded_context, encoded_query, answer)
        return output
