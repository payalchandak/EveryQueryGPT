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
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    expand_mask,
    time_from_deltas,
)


# question – is this layer called elsewhere in the codebase or can I rename to EveryQueryOutputLayer? 
# class ConditionallyIndependentGenerativeOutputLayer():
class EveryQueryOutputLayer():

    def __init__(
        self,
        config: StructuredTransformerConfig,
    ):
        super().__init__()
        self.rate_layer = torch.nn.Linear(config.hidden_size + config.query_hidden_size, 1) 
        # (todo) update config to include params

    def forward(
        self,
        encoded_context: torch.FloatTensor,
        encoded_query: torch.FloatTensor,
        answer: torch.FloatTensor,
    ) -> torch.FloatTensor:
    # ) -> GenerativeSequenceModelOutput: # question - what is the purpose of this output object
        predicted_rate = self.rate_layer(torch.cat([encoded_context, encoded_query], dim=0)) # (todo) need to check the dim
        return torch.distributions.Poisson(predicted_rate).log_prob(answer) 
        


class CIPPTForGenerativeSequenceModeling(StructuredGenerationMixin, StructuredTransformerPreTrainedModel):
    """The end-to-end model for conditionally independent generative sequence modelling.

    This model is a subclass of :class:`~transformers.StructuredTransformerPreTrainedModel` and is designed
    for generative pre-training over "event-stream" data, with inputs in the form of `PytorchBatch` objects.
    It is trained to solve the generative, multivariate, masked temporal point process problem over the
    defined measurements in the input data.

    This model largely simply passes the input data through a
    `ConditionallyIndependentPointProcessTransformer` followed by a
    `ConditionallyIndependentGenerativeOutputLayer`.

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
        self.query_encoder = None
        self.output_layer = EveryQueryOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init() # question – why call it when post init is not defined for this class ??? 

    # question – can we remove this function? 
    def prepare_inputs_for_generation(
        self, batch: PytorchBatch, past: tuple | None = None, **kwargs
    ) -> dict[str, Any]:
        """Returns model keyword arguments that have been modified for generation purposes.

        Args:
            batch: The batch of data to be transformed.
            past: The past state of the model, if any. If specified, it must be a tuple containing the past
                values over prior layers and heads.

            **kwargs: Additional keyword arguments. If "use_cache" is set in the kwargs to False, then the
                past state is ignored. If not, then the past state is passed through the model to accelerate
                generation, if past is not None then the batch is trimmed to the last element in the sequence,
                and the sequential attention mask is pre-computed.

        Raises:
            ValueError: If the past state is malformed or if there is a dep_graph_el_generation_target in the
                kwargs that is not None.
        """
        # only last sequence element in the batch if past is defined in kwargs
        batch.time = time_from_deltas(batch)

        use_cache = kwargs.get("use_cache", False)
        if not use_cache:
            return {**kwargs, "batch": batch}

        seq_attention_mask = expand_mask(batch.event_mask, batch.time_delta.dtype)

        dep_graph_el_generation_target = kwargs.get("dep_graph_el_generation_target", None)
        if dep_graph_el_generation_target is not None:
            raise ValueError(
                f"Can't use dep_graph_el_generation_target ({dep_graph_el_generation_target}) "
                "in a conditionally independent model."
            )

        match past:
            case None:
                pass

            case tuple():
                batch = batch.last_sequence_element_unsqueezed()

            case _:
                raise ValueError(f"{past} malformed!")

        return {
            **kwargs,
            "seq_attention_mask": seq_attention_mask,
            "batch": batch,
            "past": past,
        }

    def forward(self, batch: PytorchBatch, **kwargs) -> torch.FloatTensor: 
    # ) -> GenerativeSequenceModelOutput: again why this output object? 

        # use_cache = kwargs.get("use_cache", False) # question – what is the cache for? 
        # output_attentions = kwargs.get("output_attentions", False)
        # output_hidden_states = kwargs.get("output_hidden_states", False)

        context, query, answer = batch
        encoded_context = self.context_encoder(context, **kwargs)
        encoded_query = self.query_encoder(query, **kwargs)
        output = self.output_layer(encoded_context.last_hidden_state, encoded_query, answer)

        # if use_cache:
        #     output["past_key_values"] = encoded_context.past_key_values
        # if output_attentions:
        #     output["attentions"] = encoded_context.attentions
        # if output_hidden_states:
        #     output["hidden_states"] = encoded_context.hidden_states

        return output
