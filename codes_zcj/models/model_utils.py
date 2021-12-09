# coding=utf-8

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (PreTrainedTokenizer, PreTrainedModel, PretrainedConfig)


class BaseModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.toker = None
    
    def tie_tokenizer(self, toker: PreTrainedTokenizer):
        self.toker = toker
        if len(self.toker) > self.toker.vocab_size:
            self.resize_token_embeddings(len(self.toker))
