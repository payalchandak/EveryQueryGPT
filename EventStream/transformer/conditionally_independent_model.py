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

class EveryQueryOutputLayerwithPoissonLoss(torch.nn.Module):

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()
        self.proj = torch.nn.Linear(config.hidden_size * 2, 1) 
        self.objective = torch.nn.PoissonNLLLoss(log_input=True)

    def forward(
        self,
        encoded_context: torch.FloatTensor,
        encoded_query: torch.FloatTensor,
        answer: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        assert encoded_context.shape == encoded_query.shape, f"encoded_context {encoded_context.shape} and encoded_query {encoded_query.shape} should be (batch_size, hidden_size)"
        
        log_rate = self.proj(torch.cat([encoded_context, encoded_query], dim=1)) 
        loss = self.objective(log_rate, answer)
        rate = torch.exp(log_rate)
        manual_loss = torch.mean( rate - (answer * log_rate) ) 
        dloss_drate = torch.mean( 1 - ( answer / rate ) )
        
        out = {
            'loss':loss, 
            'log_rate':log_rate.squeeze(),
            'rate':rate.squeeze(),
            'answer': answer.squeeze(),
            'manual_loss':manual_loss, 
            'dloss_drate': dloss_drate.squeeze(),
            
        }
        return out 

class EveryQueryOutputLayerwithZeroBCEandTruncatedPoissonLossandPopulationRate(torch.nn.Module):

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()
        self.rate_proj = torch.nn.Linear(1, 1)
        self.zero_proj = torch.nn.Linear(1, 1) 
        self.zero_objective = torch.nn.BCEWithLogitsLoss()
    
    def stable_log_exp_minus_one_exp(self, x):
        # torch.exp(x) - torch.log(torch.special.expm1(torch.exp(x))) 
        # this difference is numerically zero at 2.683
        threshold = torch.tensor(2.6)
        large_x_approximation = torch.exp(x)
        small_x_approximation = torch.log(torch.special.expm1(torch.exp(x)))
        result = torch.where(x > threshold, large_x_approximation, small_x_approximation)
        return result

    def sample_zero_truncated_poisson(self, log_rate):
        rate = torch.exp(log_rate).squeeze()
        u = torch.distributions.uniform.Uniform(low=torch.exp(-rate), high=1).sample()
        t = -torch.log(u)
        return 1 + torch.distributions.poisson.Poisson(rate = rate-t).sample()

    def forward(
        self,
        encoded_context: torch.FloatTensor,
        encoded_query: torch.FloatTensor,
        answer: torch.FloatTensor,
        population_rate: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        assert encoded_context.shape == encoded_query.shape, f"encoded_context {encoded_context.shape} and encoded_query {encoded_query.shape} should be (batch_size, hidden_size)"
        
        zero_truth = (answer == .0).float()
        proj_inputs = torch.cat([zero_truth.unsqueeze(1)], dim=1)
        zero_logits = self.zero_proj(zero_truth.unsqueeze(1)).squeeze(dim=-1)
        # zero_truth = (answer == .0).float()
        zero_loss = self.zero_objective(zero_logits, zero_truth).mean()
        print(zero_truth, answer, zero_logits)
        print()

        # Since we have cross entropy loss for zero rates, the minimum value for rate should be 1. 
        # This means the minimum value for log_rate is 0, and thus we can put it through a ReLU. 
        log_rate = torch.log(population_rate).unsqueeze(1) + self.rate_proj(proj_inputs)
        log_rate = torch.relu(log_rate)  

        mask = (answer != .0)
        if torch.sum(mask) > 0: 
            trucated_poisson_loss = - (answer[mask] * log_rate[mask]) + self.stable_log_exp_minus_one_exp(log_rate[mask])
            trucated_poisson_loss = torch.mean(trucated_poisson_loss)
        else:
            trucated_poisson_loss = torch.zeros_like(zero_loss)

        loss = zero_loss # + trucated_poisson_loss

        zero_sample = torch.distributions.bernoulli.Bernoulli(logits=zero_logits).sample()
        rate_sample = self.sample_zero_truncated_poisson(log_rate) 
        rate = torch.where(zero_sample.bool(), torch.zeros_like(zero_sample), rate_sample)

        out = {
            'loss':loss, 
            'trucated_poisson_loss':trucated_poisson_loss, 
            'zero_loss':zero_loss, 
            'rate':rate.squeeze(),
            'answer':answer.squeeze(),
            'truncated_rate':torch.exp(log_rate)[mask].squeeze(), # corrcoef where (answer != .0)
            'truncated_answer':answer[mask].squeeze(),
            'zero_prob':torch.sigmoid(zero_logits), 
            'zero_truth':zero_truth,
        }
        return out 

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
        
        zero_logits = self.zero_proj(proj_inputs).squeeze(dim=-1)
        zero_loss = self.zero_objective(zero_logits, (answer == .0).float())
        zero_loss = torch.mean(zero_loss)

        # Since we have cross entropy loss for zero rates, the minimum value for rate should be 1. 
        # This means the minimum value for log_rate is 0, and thus we can put it through a ReLU.     
        log_rate = self.rate_proj(proj_inputs)

        mask = (answer != .0)
        if torch.sum(mask) > 0: 
            trucated_poisson_loss = - (answer[mask] * log_rate[mask]) + self.stable_log_exp_minus_one_exp(log_rate[mask])
            trucated_poisson_loss = torch.mean(trucated_poisson_loss)
        else:
            trucated_poisson_loss = 0 

        loss = zero_loss + trucated_poisson_loss

        zero_prob = torch.sigmoid(zero_logits)
        rate = torch.where(zero_prob >= 0.5, torch.zeros_like(zero_prob), torch.exp(log_rate).squeeze())
        
        out = {
            'loss':loss, 
            'trucated_poisson_loss':trucated_poisson_loss, 
            'zero_loss':zero_loss, 
            'rate':rate.squeeze(),
            'answer': answer.squeeze(),
        }
        return out 
 
# class CIPPTForGenerativeSequenceModeling(StructuredTransformerPreTrainedModel): 
#     def __init__(
#         self,
#         config: StructuredTransformerConfig,
#     ):
#         super().__init__(config)

#         if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
#             raise ValueError(f"{config.structured_event_processing_mode} invalid!")

#         self.context_encoder = ConditionallyIndependentPointProcessTransformer(config)
#         self.query_embedding_layer = self.context_encoder.input_layer.data_embedding_layer.query_embedding
#         self.query_encoder = None # MLP ?? 
#         self.output_layer = EveryQueryOutputLayerwithZeroBCEandTruncatedPoissonLossandPopulationRate(config)
    
#     def safe_max_seq_dim(self, X: torch.Tensor, mask: torch.BoolTensor):
#         # X is batch_size, seq_len, hidden_dim
#         mask = mask.unsqueeze(-1).expand_as(X) 
#         masked_X = torch.where(mask, X, -float("inf"))
#         maxes = masked_X.max(1)[0]
#         return torch.nan_to_num(maxes, nan=None, posinf=None, neginf=0)

#     def forward(self, batch: PytorchBatch, **kwargs) -> torch.FloatTensor: 
#         context, query, answer = batch['context'], batch['query'], batch['answer']
#         encoded_context = self.context_encoder(context, **kwargs)
#         encoded_context = self.safe_max_seq_dim(encoded_context.last_hidden_state, context.event_mask)
#         query_embed = self.query_embedding_layer(query, **kwargs) 
#         encoded_query = query_embed # (todo) self.query_encoder(query_embed, **kwargs)
#         output = self.output_layer(encoded_context, encoded_query, answer, query['population_rate'])
#         return output

class CIPPTForGenerativeSequenceModeling(StructuredTransformerPreTrainedModel): 
    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__(config)

        if config.structured_event_processing_mode != StructuredEventProcessingMode.CONDITIONALLY_INDEPENDENT:
            raise ValueError(f"{config.structured_event_processing_mode} invalid!")

        self.context_encoder = ConditionallyIndependentPointProcessTransformer(config)
        self.query_embedding_layer = self.context_encoder.input_layer.data_embedding_layer.query_embedding
        self.query_encoder = None # MLP ?? 
        self.output_layer = EveryQueryOutputLayerwithZeroBCEandTruncatedPoissonLossandPopulationRate(config)
    
        self.zero_proj = torch.nn.Linear(config.hidden_size*2 + 1, 1) 
        self.zero_objective = torch.nn.BCEWithLogitsLoss()
    
    def safe_max_seq_dim(self, X: torch.Tensor, mask: torch.BoolTensor):
        # X is batch_size, seq_len, hidden_dim
        mask = mask.unsqueeze(-1).expand_as(X) 
        masked_X = torch.where(mask, X, -float("inf"))
        maxes = masked_X.max(1)[0]
        return torch.nan_to_num(maxes, nan=None, posinf=None, neginf=0)


    def forward(self, batch: PytorchBatch, **kwargs) -> torch.FloatTensor: 
        context, query, answer = batch['context'], batch['query'], batch['answer']

        encoded_context = self.context_encoder(context, **kwargs)
        encoded_context = self.safe_max_seq_dim(encoded_context.last_hidden_state, context.event_mask)
        query_embed = self.query_embedding_layer(query, **kwargs) 
        encoded_query = query_embed 
        
        zero_truth = (answer == 0).float()
        zero_logits = self.zero_proj(torch.cat([encoded_context, encoded_query, zero_truth.unsqueeze(1)],dim=1)).squeeze(dim=-1)
        zero_prob = torch.sigmoid(zero_logits)

        print('grad',[param.grad for param in self.zero_proj.parameters()])
        print('param',[param for param in self.zero_proj.parameters()])
        zero_loss = self.zero_objective(zero_logits, zero_truth).mean()

        output = {
            'loss':zero_loss, 
            'zero_loss':zero_loss, 
            'zero_prob':zero_prob, 
            'zero_truth':zero_truth,
            'rate':torch.tensor([]),
            'answer':torch.tensor([]),
            'truncated_rate':torch.tensor([]),
            'truncated_answer':torch.tensor([]),
        }
        return output
