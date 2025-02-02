import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
import math
from functools import partial
from dataclasses import dataclass
from beartype.typing import List, Optional
from utils import (all_rows_have_eos_id, append_eos_id,
                                                batch_unique_consecutive, beartype_jit, ceil_div, default,
                                                eval_decorator, exists, float32_to_int16,
                                                generate_mask_with_prob, get_embeds, gumbel_sample,
                                                int16_to_float32, mask_out_after_eos_id,
                                                round_down_nearest_multiple, top_k)
from .transformers import Transformer
from tqdm import tqdm
import itertools

@dataclass
class TokenSequenceInfo():
    """
    Defines a token sequence to be conditioned on or predicted in TokenConditionedTransformer
    """
    codebook_size: int
    num_quantizers: int    # e.g. 1 for semantic, Q for coarse acoustic, ...
    unique_consecutive: bool 

class TokenConditionedTransformer(nn.Module):
    """
    Combination of the SemanticTransformer, CoarseTransformer and FineTransformer in lucidrain's AudioLM implementation.
    Can handle a variable number of token sequences, each with their own parameters.
    https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/audiolm_pytorch.py
    """
    # TODO: Add in text conditioning for parity with AudioLM. Not important for MusicLM though.

    def __init__(
        self,
        *,
        token_sequences: List[TokenSequenceInfo],
        dim,
        depth,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.1,
        has_condition=False,
        cond_as_self_attn_prefix=False,
        cond_drop_prob=0.5,
        grad_shrink_alpha=0.1,
        use_absolute_position_embeddings=False,
        max_absolute_position_embeddings=262,
        **kwargs
    ):
        super().__init__()

        self.token_sequences = token_sequences

        self.has_condition = has_condition
        self.cond_drop_prob = cond_drop_prob
        self.use_absolute_position_embeddings = use_absolute_position_embeddings

        self.start_tokens = torch.nn.ParameterList()
        self.logit_weights = torch.nn.ParameterList()
        self.embeddings = torch.nn.ModuleList()
        self.absolute_position_embeddings = torch.nn.ModuleList() if self.use_absolute_position_embeddings else None
        self.eos_ids = []

        for sequence in token_sequences:
            self.start_tokens.append(nn.Parameter(torch.randn(dim)))
            self.eos_ids.append(sequence.codebook_size)

            codebook_size_with_eos = sequence.codebook_size + 1

            self.embeddings.append(nn.Embedding(codebook_size_with_eos * sequence.num_quantizers, dim))
            self.logit_weights.append(nn.Parameter(torch.randn(sequence.num_quantizers, codebook_size_with_eos, dim)))

            if self.use_absolute_position_embeddings:
                self.absolute_position_embeddings.append(nn.Embedding(max_absolute_position_embeddings, dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            cross_attend=has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix=cond_as_self_attn_prefix,
            grad_shrink_alpha=grad_shrink_alpha,
            **kwargs
        )
        # self.device
    @property
    def device(self):
        return "cuda"
    #     return next(self.parameters()).device

    def forward(self,
                *,
                all_token_ids: List[torch.Tensor],
                self_attn_mask=None,
                cond_drop_prob=None,
                return_only_final_seq_logits=False
                ):
        """
        all_token_ids: List of tensors containing token ids. Each element can either be 2 dimensional (batch_size, n_time_steps * num_quantizers) or 3 dimensional (batch_size, n_time_steps, num_quantizers)
                       Each element in list corresponds to one token sequence in self.token_sequences (e.g. semantic, coarse acoustic, fine acoustic, etc.)

        return_only_final_seq_logits: If True, only return logits for the final token sequence in self.token_sequences.
        """

        b, device = all_token_ids[0].shape[0], self.device

        all_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids))

        assert len(all_token_ids) == len(self.token_sequences) == len(self.embeddings)

        tokens = []
        start_tokens = []
        split_at = []
        for idx, sequence, token_ids, embedding, start_token in zip(range(len(self.token_sequences)), self.token_sequences, all_token_ids, self.embeddings, self.start_tokens):

            # add offsets
            if sequence.num_quantizers > 1:
                offsets = sequence.codebook_size * torch.arange(sequence.num_quantizers, device=device)
                offsets = repeat(offsets, 'q -> 1 (n q)', n=ceil_div(token_ids.shape[-1], sequence.num_quantizers))
                offsets = offsets[:, :token_ids.shape[-1]]
                token_ids = token_ids + offsets

            # get embeddings and prepare for next step
            token_embeddings = get_embeds(embedding, token_ids, pad_id=-1)
            if self.use_absolute_position_embeddings:
                position_embeddings = self.absolute_position_embeddings[idx](torch.arange(token_embeddings.shape[1], device=device)[None, :])
                tokens.append(token_embeddings + position_embeddings)
            else:
                tokens.append(token_embeddings)
            start_tokens.append(repeat(start_token, 'd -> b 1 d', b=b))

            n_tokens = token_embeddings.shape[1] + 1 # +1 for start token of next sequence
            split_at.append(n_tokens if len(split_at) == 0 else split_at[-1] + n_tokens)

        tokens = list(itertools.chain(*zip(start_tokens, tokens)))  # [start_1, tokens_1, start_2, tokens_2, ...]
        tokens = torch.cat(tokens, dim=1)

        tokens = self.transformer(tokens, self_attn_mask=self_attn_mask)

        split_at = split_at[:-1]  # remove last element (total number of tokens)
        all_pred_tokens = torch.tensor_split(tokens, split_at, dim=1)

        # strip next start token from end of every sequence besides last
        # in tokens: s1 t1 t2 t3 t4 .. e1   s2 t1 t2 t3 t4 e2
        # out logit: t1 t2 t3 t4 .. e1 s2   t1 t2 t3 t4 e2
        # split:    [t1 t2 t3 t4 .. e1 s2] [t1 t2 t3 t4 e2]
        all_pred_tokens = [pred_tokens[:, :-1] for pred_tokens in all_pred_tokens[:-1]] + [all_pred_tokens[-1]]

        # get logits

        all_logits = []
        assert len(all_pred_tokens) == len(self.token_sequences) == len(self.logit_weights)

        for index, (sequence, pred_tokens, seq_logit_weights) in enumerate(zip(self.token_sequences, all_pred_tokens, self.logit_weights)):
            if not return_only_final_seq_logits or index == len(self.token_sequences) - 1:
                n = pred_tokens.shape[1]
                nq = round_down_nearest_multiple(n, sequence.num_quantizers)

                pred_tokens_groupable, pred_tokens_remainder = pred_tokens[:, :nq], pred_tokens[:, nq:]

                pred_tokens_groupable = rearrange(
                    pred_tokens_groupable, 'b (n q) d -> b n q d', q=sequence.num_quantizers)

                pred_logits_groupable = einsum('q c d, b n q d -> b n q c', seq_logit_weights, pred_tokens_groupable)

                pred_logits_groupable = rearrange(pred_logits_groupable, 'b n q c -> b (n q) c')

                remainder_num_tokens_in_step = pred_tokens_remainder.shape[1]

                if remainder_num_tokens_in_step > 0:
                    pred_logits_remainder = einsum(
                        'q c d, b q d -> b q c', seq_logit_weights[:remainder_num_tokens_in_step], pred_tokens_remainder)
                    pred_logits = torch.cat((pred_logits_groupable, pred_logits_remainder), dim=1)
                else:
                    pred_logits = pred_logits_groupable

                all_logits.append(pred_logits)
            else:
                all_logits.append(None)

        return all_logits
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=3,
        **kwargs
    ):
        """Doesn't do anything without the AudioLM-pytorch text conditioning implementation. Just use forward() instead."""

        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1 or not self.has_condition:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)

        scaled_logits = []

        for seq_logits, null_seq_logits in zip(logits, null_logits):
            if seq_logits is None:
                scaled_logits.append(None)
            else:
                scaled_logits.append(null_seq_logits + (seq_logits - null_seq_logits) * cond_scale)

        return scaled_logits

@beartype_jit
class TokenConditionedTransformerWrapper(nn.Module):
    """Combination of SemanticTransformerWrapper, CoarseTransformerWrapper and FineTransformerWrapper in lucidrain's audiolm-pytorch, without the input processing + text conditioning"""
    def __init__(
        self,
        *,
        transformer: TokenConditionedTransformer,
        pad_id=-1,
        unique_consecutive=True,
        cross_entropy_loss_weights: Optional[List[float]] = None,
        mask_prob=0.15
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer = torch.nn.DataParallel(self.transformer)

        self.token_sequences = transformer.token_sequences

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.cross_entropy_loss_weights = default(cross_entropy_loss_weights, [1 for _ in self.token_sequences])

        self.eos_ids = transformer.eos_ids
        self.mask_prob = mask_prob
        assert len(self.token_sequences) == len(self.eos_ids) == len(self.cross_entropy_loss_weights)

    @property
    def device(self):
        # return next(self.parameters()).device
        return "cuda"

    @eval_decorator
    @torch.no_grad()
    @beartype_jit
    def generate(
        self,
        *,
        conditioning_token_ids: List[torch.Tensor],
        pred_token_ids: Optional[torch.Tensor] = None,
        max_time_steps=1024,
        filter_thres=0.9,
        temperature=1.,
        include_eos_in_output=False,
        append_eos_to_conditioning_tokens=True,
        allow_eos_in_output=False,
        **kwargs
    ):
        assert len(conditioning_token_ids) == len(self.token_sequences) - 1

        batch, device = conditioning_token_ids[0].shape[0], self.device

        conditioning_token_ids = [t.to(device) for t in conditioning_token_ids]

        if exists(pred_token_ids):
            assert pred_token_ids.shape[0] == batch
            init_pred_time_step = pred_token_ids.shape[1]
            pred_token_ids = rearrange(pred_token_ids, 'b ... -> b (...)')
        else:
            init_pred_time_step = 0
            pred_token_ids = torch.empty((batch, 0), device=device, dtype=torch.long)

        pred_sequence_info, pred_eos_id = self.token_sequences[-1], self.eos_ids[-1]

        # batch unique consecutive
        for index, sequence_info in enumerate(self.token_sequences[:-1]):
            if sequence_info.unique_consecutive:
                conditioning_token_ids[index] = batch_unique_consecutive(
                    conditioning_token_ids[index], pad_value=self.pad_id)
        if self.token_sequences[-1].unique_consecutive:
            pred_token_ids = batch_unique_consecutive(pred_token_ids, pad_value=self.pad_id)

        # reshape and append eos
        if append_eos_to_conditioning_tokens:
            conditioning_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), conditioning_token_ids))
            conditioning_token_ids = [append_eos_id(ids, eos_id) for ids, eos_id in zip(conditioning_token_ids, self.eos_ids)]

        # initialize

        sampled_pred_token_ids = pred_token_ids.clone()

        for time_step in tqdm(range(init_pred_time_step, max_time_steps), desc='generating predicted tokens'):
            for ind in range(pred_sequence_info.num_quantizers):
                is_last_step = ind == (pred_sequence_info.num_quantizers - 1)

                pred_logits = self.transformer(
                    all_token_ids=conditioning_token_ids + [sampled_pred_token_ids],
                    return_only_final_seq_logits=True,
                    **kwargs
                )[-1]

                last_pred_logits = pred_logits[:, -1]

                if not allow_eos_in_output or not is_last_step:
                    # prevent eos 1) if we don't allow it or 2) in the middle of a time step
                    last_pred_logits[:, -1] = float('-inf')

                filtered_logits = top_k(last_pred_logits, thres=filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_pred_token_ids = torch.cat((sampled_pred_token_ids, sampled), dim=-1)

        sampled_pred_token_ids = mask_out_after_eos_id(
            sampled_pred_token_ids, pred_eos_id, keep_eos=include_eos_in_output)
        sampled_pred_token_ids = rearrange(
            sampled_pred_token_ids, 'b (n q) -> b n q', q= pred_sequence_info.num_quantizers)

        return sampled_pred_token_ids

    def forward(
        self,
        *,
        all_token_ids: List[torch.Tensor],
        return_loss: bool=False,
        input_has_eos: bool=False,
        **kwargs
    ):
        assert len(all_token_ids) == len(self.token_sequences)

        batch, device = all_token_ids[0].shape[0], self.device

        all_token_ids = list(map(lambda t: rearrange(t, 'b ... -> b (...)'), all_token_ids))

        if self.training:
            assert not input_has_eos, "train sequences (from clap, wav2vec, etc.) shouldn't come with an eos token"

        # append eos to sequences if not already there
        if not input_has_eos:
            all_token_ids = [append_eos_id(ids, eos_id) for ids, eos_id in zip(all_token_ids, self.eos_ids)]

        if self.unique_consecutive:
            for index, sequence_info in enumerate(self.token_sequences):
                if sequence_info.unique_consecutive:
                    all_token_ids[index] = batch_unique_consecutive(all_token_ids[index], pad_value=self.pad_id)

        if return_loss:
            all_labels = [ids.clone() for ids in all_token_ids]
            all_token_ids[-1] = all_token_ids[-1][:, :-1]  # don't include last token when returning loss (should be eos)

        # do not attend to padding tokens or eos tokens
        combined_self_attn_mask = torch.empty((batch, 0), device=device, dtype=torch.bool)
        for ids, eos_id in zip(all_token_ids[:-1], self.eos_ids[:-1]):
            mask = (ids != self.pad_id) & (ids != eos_id)

            ids.masked_fill_(~mask, 0)  # inplace

            # transformer appends a start token to beginning of sequence, so add to mask
            mask = F.pad(mask, (1, 0), value=True)
            combined_self_attn_mask = torch.cat((combined_self_attn_mask, mask), dim=-1)

        # add our predicted tokens + start token to our mask
        pred_token_len = all_token_ids[-1].shape[-1]
        combined_self_attn_mask = F.pad(combined_self_attn_mask, (0, pred_token_len + 1), value=True)

        # forgetful causal mask - structured dropout
        if self.mask_prob > 0 and self.training:
            combined_self_attn_mask &= generate_mask_with_prob(
                combined_self_attn_mask.shape, self.mask_prob, device=combined_self_attn_mask.device)

        all_logits = self.transformer(
            all_token_ids=all_token_ids,
            self_attn_mask=combined_self_attn_mask,
            **kwargs
        )

        # whether to early return the logits

        if not return_loss:
            return all_logits

        all_logits = list(map(lambda t: rearrange(t, 'b n c -> b c n'), all_logits))

        total_logits = 0
        running_loss = 0.
        for logits, labels, cross_entropy_loss_weight, sequence_info in zip(all_logits, all_labels, self.cross_entropy_loss_weights, self.token_sequences):
            loss = 0.
            num_logits = 0
            unique_consecutive = sequence_info.unique_consecutive and self.unique_consecutive

            if cross_entropy_loss_weight > 0 and exists(logits):
                num_logits = (labels != self.pad_id).sum() if unique_consecutive else labels.numel()

                loss = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=self.pad_id if unique_consecutive else -100
                )

            total_logits += num_logits
            running_loss += loss * num_logits * cross_entropy_loss_weight

        return running_loss / total_logits, all_logits, all_labels

# class TokenConditionedTransformer:
#     pass 

# class TokenConditionedTransformerWrapper:
#     pass