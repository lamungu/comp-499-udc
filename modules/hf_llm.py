
import logging

from torch import Tensor
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class HuggingFaceLLM(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained LLM model.
      Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "gpt2"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    """

    def __init__(self, source: str, save_path: str, freeze: bool = False) -> None:
        super().__init__()
        self.freeze = freeze
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            cache_dir=save_path
        )
        if self.freeze:
            logger.warning(
                "[HuggingFaceLLM] - LLM Model is frozen."
            )
            # We keep it to train() to have dropout and LN computed adequaly
            self.model.train()  
            for param in self.model.parameters():
                param.requires_grad = False

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, output_hidden_states = False):
        with torch.set_grad_enabled(not self.freeze):
            output = self.model.forward(
                input_ids,
                token_type_ids=token_type_ids,
                output_hidden_states=output_hidden_states
            )
        return output
