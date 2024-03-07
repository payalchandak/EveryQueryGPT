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

class EveryQueryOutputLayerwithZeroBCEandTruncatedPoissonLoss(torch.nn.Module):

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()
        self.rate_proj = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size*2, 1),
            torch.nn.ReLU(),
        )
        self.zero_proj = torch.nn.Linear(config.hidden_size * 2, 1) 
        self.zero_objective = torch.nn.BCEWithLogitsLoss()
        # (todo) update config to include separate query_hidden_size 
    
    def stable_log_exp_minus_one_exp(self, x):
        # torch.exp(x) - torch.log(torch.special.expm1(torch.exp(x))) 
        # this difference is numerically zero at 2.683
        threshold = torch.tensor(2.6)
        large_x_approximation = torch.exp(x)
        small_x_approximation = torch.log(torch.special.expm1(torch.exp(x)))
        result = torch.where(x > threshold, large_x_approximation, small_x_approximation)
        return result
    
    def forward(
        self,
        encoded_context: torch.FloatTensor,
        encoded_query: torch.FloatTensor,
        answer: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        assert encoded_context.shape == encoded_query.shape, f"encoded_context {encoded_context.shape} and encoded_query {encoded_query.shape} should be (batch_size, hidden_size)"
        
        proj_inputs = torch.cat([encoded_context, encoded_query], dim=1)
        
        zero_mask = (answer == .0).float()
        zero_logits = self.zero_proj(proj_inputs).squeeze(dim=-1)
        zero_loss = self.zero_objective(zero_logits, zero_mask)
        zero_loss = torch.mean(zero_loss)

        non_zero_mask = (answer != .0)
        non_zero_answer = answer[non_zero_mask]
        # Since we have cross entropy loss, the minimum value for rate should be 1. 
        # This means the minimum value for log_rate is 0, and thus we put it through a ReLU. 
        log_rate = self.rate_proj(proj_inputs[non_zero_mask]) 
        rate = torch.exp(log_rate) 
        trucated_poisson_loss = - (non_zero_answer * log_rate) + self.stable_log_exp_minus_one_exp(log_rate)
        trucated_poisson_loss = torch.mean(trucated_poisson_loss)
        loss = zero_loss + trucated_poisson_loss
        
        out = {
            'loss':loss, 
            'trucated_poisson_loss':trucated_poisson_loss, 
            'zero_loss':zero_loss, 
            'num_pos_rate':torch.sum(non_zero_mask),
            'log_rate':log_rate.squeeze(),
            'rate':rate.squeeze(),
            'answer': non_zero_answer.squeeze(),
        }
        return out 
        

class CIPPTForGenerativeSequenceModeling(StructuredTransformerPreTrainedModel): 
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.context_encoder = ConditionallyIndependentPointProcessTransformer(config)
        # self.separate_query_embedding_layer = DataEmbeddingLayer(
        #     n_total_embeddings=config.vocab_size,
        #     out_dim=config.hidden_size,
        #     categorical_embedding_dim=config.categorical_embedding_dim,
        #     numerical_embedding_dim=config.numerical_embedding_dim,
        #     static_embedding_mode=config.static_embedding_mode,
        #     split_by_measurement_indices=None,
        #     do_normalize_by_measurement_index=config.do_normalize_by_measurement_index,
        #     static_weight=config.static_embedding_weight,
        #     dynamic_weight=config.dynamic_embedding_weight,
        #     categorical_weight=config.categorical_embedding_weight,
        #     numerical_weight=config.numerical_embedding_weight,
        # )
        # self.query_embedding_layer = self.separate_query_embedding_layer.query_embedding
        # add trainer strategy ddp find unused true in config if using separate embedding layers 
        self.query_embedding_layer = self.context_encoder.input_layer.data_embedding_layer.query_embedding
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
