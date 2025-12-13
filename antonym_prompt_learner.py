import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class PromptLearner(nn.Module):
    def __init__(self, tokenizer, text_encoder, quality_type, class_token_position = 'end'):
        super().__init__()
        n_ctx = 16
        
        dtype = text_encoder.dtype
        ctx_dim = text_encoder.text_model.final_layer_norm.weight.shape[0]
        #clip_imsize = clip_model.visual.input_resolution
        


        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f'Context Position: {class_token_position}')

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        name = quality_type[0]
        self.name_len = len(name.split(' '))

        prompts = [prompt_prefix + " " + quality_type[0] + ".", prompt_prefix + " " + quality_type[1] + "."]

        tokenized_prompts = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(text_encoder.device)
        
        attention_mask = tokenized_prompts['attention_mask']
        with torch.no_grad():
            embedding = text_encoder.text_model.embeddings.token_embedding(tokenized_prompts['input_ids']).to(text_encoder.device)
            #embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.class_token_position = class_token_position
        self.attention_mask = attention_mask

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).repeat(2,1,1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.class_token_position == 'end':
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        elif self.class_token_position == 'front':
            name_len = self.name_len
            prompts = torch.cat(
                [
                    prefix,
                    suffix[:, :name_len, :],
                    ctx,
                    suffix[:, name_len: , :]
                ], 
                dim = 1,
            )

        return prompts
    





class ModifiedTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder.text_model.encoder
        self.positional_embedding = text_encoder.text_model.embeddings.position_embedding
        self.final_layer_norm = text_encoder.text_model.final_layer_norm
        #self.text_projection = clip_model.text_projection
        self.dtype = text_encoder.dtype
        self.register_buffer("position_ids", torch.arange(77).expand((1, -1)))

    def forward(self, prompts, tokenized_prompts, attention_mask):

        input_shape = prompts.shape[:2]
        #input_ids = input_ids.view(-1, input_shape[-1])
        bsz, seq_len = input_shape
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.positional_embedding(position_ids)
        hidden_states = prompts + position_embeds


        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype).to(hidden_states.device)
        

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        '''pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]'''

        return last_hidden_state

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask



class CustomCLIP(nn.Module):
    def __init__(self, tokenizer, text_encoder, quality_type, class_token_position):
        super().__init__()
        self.prompt_learner = PromptLearner(tokenizer, text_encoder, quality_type, class_token_position)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.attention_mask = self.prompt_learner.attention_mask
        #self.image_encoder = clip_model.visual
        self.txt_encoder = ModifiedTextEncoder(text_encoder)
        #self.logit_scale = clip_model.logit_scale
        #self.dtype = clip_model.dtype

    def forward(self):

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        attention_mask = self.attention_mask
        text_features = self.txt_encoder(prompts, tokenized_prompts, attention_mask)

        return text_features
    
