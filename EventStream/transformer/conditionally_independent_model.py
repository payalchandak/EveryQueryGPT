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


class EQTForPretraining(StructuredTransformerPreTrainedModel):
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

        self.encoder = ConditionallyIndependentPointProcessTransformer(config)
        self.query_encoder = ???(config)
        self.output_layer = ConditionallyIndependentGenerativeOutputLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, batch: PytorchBatch, query_batch: Any, answer: Any, **kwargs
    ) -> EveryQueryModelOutput:
        """This runs the full forward pass of the model.

        Args:
            batch: The batch of data to be transformed.
            TODO
            **kwargs: Additional keyword arguments, which are used for output structuring and are forwarded to
                the encoder. The model specifically looks for use_cache, output_attentions, and
                output_hidden_states keyword arguments, which control whether additional properties should be
                added to the output.

        Returns:
            The output of the model, which is a `GenerativeSequenceModelOutput` object.
        """
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        encoded = self.encoder(batch, **kwargs)
        query_encoded = self.query_encoder(query_batch, **kwargs)

        query_contextualized_encoding = self.query_contextualizer(encoded, query_encoded)

    # Rate layer helpers
    #     class ExponentialTTELayer(torch.nn.Module):
    # """A class that outputs an exponential distribution for time-to-event.

    # This class is used to initialize the ExponentialTTELayer and project the input tensor to get the
    # implied exponential distribution.

    # Args:
    #     in_dim: The dimensionality of the input.
    # """

    # def __init__(self, in_dim: int):
    #     super().__init__()
    #     self.proj = torch.nn.Linear(in_dim, 1)

    # def forward(self, T: torch.Tensor) -> torch.distributions.exponential.Exponential:
    #     """Forward pass.

    #     Args:
    #         T: The input tensor.

    #     Returns:
    #         An `Exponential` distribution with parameters specified by `self.proj(T)` which has output shape
    #         `(batch_size, sequence_length, 1)`.
    #     """
    #     # torch.nn.functional.elu has Image (-1, 1), but we need our rate parameter to be > 0. So we need to
    #     # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
    #     # by the smallest possible positive value permissible given the type of `T`.
    #     rate = torch.nn.functional.elu(self.proj(T)) + 1 + torch.finfo(T.dtype).tiny

    #     # The rate currently has shape (batch_size, sequence_length, 1). We want to squeeze that last
    #     # dimension.
    #     rate = rate.squeeze(dim=-1)

    #     return torch.distributions.exponential.Exponential(rate=rate)

        output = poisson(self.rate_layer(query_context...)).pdf(answer)

        if use_cache:
            output["past_key_values"] = encoded.past_key_values

        if output_attentions:
            output["attentions"] = encoded.attentions

        if output_hidden_states:
            output["hidden_states"] = encoded.hidden_states

        return output
