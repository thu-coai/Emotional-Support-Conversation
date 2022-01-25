# coding=utf-8
# copied from gpt2

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.gpt2 import (GPT2Config, GPT2LMHeadModel,)
from transformers.modeling_outputs import (CausalLMOutputWithCrossAttentions,)
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self.toker is not None
        
        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None) == (decoder_input_ids is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation: # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if decoder_input_ids is not None:
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                input_ids = torch.cat([input_ids, decoder_input_ids], dim=-1)
                labels = torch.cat([-100 * labels.new_ones(attention_mask.size()), labels], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(decoder_input_ids.size())], dim=-1)
                use_cache = False
            else:
                transformer_outputs = self.transformer(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                past_key_values = transformer_outputs[1]
                input_ids = decoder_input_ids
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(input_ids.size())], dim=-1)
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation: # inference
            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )

        elif self.training: # training
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else: # validation
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, -1, -8:]
    
        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)
        
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })
    
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        **kwargs
    ):
        assert not self.training
        assert self.toker is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        
        input_ids = torch.cat([input_ids, decoder_input_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(decoder_input_ids.size())], dim=-1)
        
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        self.predict_strategy(lm_logits, encoded_info)
        
        input_ids = torch.cat([input_ids, encoded_info['pred_strat_id'].unsqueeze(-1) + len(self.toker) - 8], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(decoder_input_ids.size())], dim=-1)
        
        assert 'min_length' in kwargs and 'max_length' in kwargs
        kwargs['min_length'] = kwargs['min_length'] + input_ids.size(1)
        kwargs['max_length'] = kwargs['max_length'] + input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return encoded_info, generations[:, input_ids.size(1):]