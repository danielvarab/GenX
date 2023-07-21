import torch
import nltk
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartForConditionalGeneration, BartTokenizer
from accelerate import Accelerator
from typing import List

def _lazy_reorder_cache(past, best_index, length):
    """closure implementation that lazily evaluates the memory intensive 'index_select' operation"""
    def _inner():
        bz = past[0][0].shape[0]
        repeated_best_index = torch.tensor([best_index] * bz).to(past[0][0].device)

        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, repeated_best_index)[:, :, :length, :]
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past
    return _inner

def _reorder_cache(past, best_index, length):
    """eargerly extract candidate from cache"""
    bz = past[0][0].shape[0]
    best_index = torch.tensor([best_index] * bz).to(past[0][0].device)

    reordered_past = ()
    for layer_past in past:
        reordered_past += (
            tuple(
                past_state.index_select(0, best_index)[:, :, :length, :]
                for past_state in layer_past[:2]
            )
            + layer_past[2:],
        )
    return reordered_past


class ModelBase:
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(self, checkpoint: str, use_fp16: int = False, gpu_rank: int = 0, mixed_precision: bool = True):
        dtype = torch.float16 if use_fp16 else torch.float32
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=dtype)

        self.device = torch.device(f"cuda:{gpu_rank}")
        model = model.eval().to(self.device)
        if (not use_fp16) and mixed_precision:
            accelerator = Accelerator(mixed_precision="fp16", device_placement=False)
            model = accelerator.prepare_model(model)

        return self(tokenizer, model)

class Baseline(ModelBase):
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(self, text: str, sentences: list[str], alpha: int = 1, max_length: int = 3) -> dict:
        k = len(sentences)
        
        src_batch = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        src_batch = {k: v.to(self.device) for k, v in src_batch.items()}

        encoder = self.model.get_encoder()
        encoder_states = encoder(**src_batch).last_hidden_state
        encoder_states = encoder_states.expand(k, *encoder_states.shape[1:])

        selected = []
        accumulative_scores = []
        accumulative_indices = []

        for step in range(max_length):
            prefix = " ".join(sentences[i] for i in selected)
            if prefix: # not empty
                step_candidates = [prefix + " " + s for s in sentences]
            else:  # empty
                step_candidates = sentences

            step_candidates = [self.tokenizer.bos_token+s for s in step_candidates]

            candidate_batch = self.tokenizer(step_candidates, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)
            candidate_batch = {k: v.to(self.device) for k, v in candidate_batch.items()}

            candidate_ids = candidate_batch["input_ids"]
            candidate_lengths = candidate_batch["attention_mask"].sum(-1)

            accumulative_indices.append(candidate_ids.cpu().numpy())

            prefix_token_id = self.model.config.decoder_start_token_id
            decoder_ids = shift_tokens_right(candidate_ids, self.tokenizer.pad_token_id, prefix_token_id)


            step_output = self.model(
                encoder_outputs=(encoder_states,),
                decoder_input_ids=decoder_ids)

            logprobs = torch.log_softmax(step_output.logits, -1)
            token_scores = torch.gather(logprobs, -1, candidate_ids.unsqueeze(-1)).squeeze()
            raw_sentence_scores = (token_scores * candidate_batch["attention_mask"]).sum(-1)
            sentence_scores = raw_sentence_scores / (candidate_lengths ** alpha)

            sentence_scores[selected] = -float("inf")  # block previously selected sentences

            accumulative_scores.append(sentence_scores.cpu().numpy())

            best_idx = sentence_scores.argmax()
            selected.append(best_idx.item())

        return {
            "selected": selected,
            "summary": "\n".join(sentences[i] for i in selected),
            "scores": accumulative_scores,
            "accumulative_indices": accumulative_indices
        }

class Cached(ModelBase):
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(self, text: str, sentences: list[str], alpha: int = 1, max_length: int = 3) -> dict:
        k = len(sentences)
        prefix_token_id = self.model.config.decoder_start_token_id
        cache = None
        
        src_batch = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        src_batch = {k: v.to(self.device) for k, v in src_batch.items()}

        bos_senteces = [self.tokenizer.bos_token+s for s in sentences]
        candidate_batch = self.tokenizer(bos_senteces, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)
        candidate_batch = {k:v.to(self.device) for k, v in candidate_batch.items()}

        prefix_candidate_batch = self.tokenizer(sentences, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, add_prefix_space=True)
        prefix_candidate_batch = {k:v.to(self.device) for k, v in prefix_candidate_batch.items()}

        encoder = self.model.get_encoder()
        encoder_states = encoder(**src_batch).last_hidden_state
        encoder_states = encoder_states.expand(k, *encoder_states.shape[1:])        

        selected = []
        accumulative_scores = []
        accumulative_indices = []
        accumulative_prefix_token_ids = []
        best_score, best_length = 0.0, 0.0

        for step in range(max_length):
            if step == 0:
                labels = candidate_batch["input_ids"]
                mask = candidate_batch["attention_mask"]
            else:
                labels = prefix_candidate_batch["input_ids"]
                mask = prefix_candidate_batch["attention_mask"]

            accumulative_indices.append(labels.cpu().numpy())

            lengths = mask.sum(-1)
            extended_lengths = (best_length + lengths).long()

            decoder_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id, prefix_token_id)

            step_output = self.model(
                encoder_outputs=(encoder_states,),
                decoder_input_ids=decoder_ids,
                # @djam: as far as I can see the attention mask does nothing due to it being causal
                # decoder_attention_mask=torch.cat([past_mask, mask], -1) if step > 0 else mask,
                past_key_values=cache,
                use_cache=True)

            logprobs = torch.log_softmax(step_output.logits, -1)
            token_scores = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
            raw_sentence_scores = (token_scores * mask).sum(-1)
            raw_sentence_scores = best_score + raw_sentence_scores
            sentence_scores = raw_sentence_scores / (extended_lengths ** alpha)

            sentence_scores[selected] = -float("inf")  # block previously selected sentences

            accumulative_scores.append(sentence_scores.cpu().numpy())

            best_idx = sentence_scores.argmax()
            best_score = raw_sentence_scores[best_idx]
            best_length = extended_lengths[best_idx]

            selected.append(best_idx.item())

            # update states
            cache = step_output["past_key_values"]
            cache = _reorder_cache(cache, best_idx, extended_lengths[best_idx])
            prefix_token_id = labels[best_idx, lengths[best_idx]-1]

            accumulative_prefix_token_ids.append(prefix_token_id)

            ## > not needed as far as I understand
            # past_mask = mask[best_idx, :lengths[best_idx]].unsqueeze(0)
            # past_mask = past_mask.expand(k, *past_mask.shape[1:])

        return {
            "selected": selected,
            "summary":"\n".join(sentences[i] for i in selected),
            "scores": accumulative_scores,
            "accumulative_indices": accumulative_indices,
            "accumulative_prefix_token_ids": accumulative_prefix_token_ids
        }


class BeamSearch(ModelBase):
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(self, text: str, sentences: List[str], alpha: int = 1, max_length: int = 3, beam_size=1) -> dict:
        k = len(sentences)
        
        src_batch = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        src_batch = {k: v.to(self.device) for k, v in src_batch.items()}

        bos_senteces = [self.tokenizer.bos_token+s for s in sentences]
        candidate_batch = self.tokenizer(bos_senteces, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)
        candidate_batch = {k:v.to(self.device) for k, v in candidate_batch.items()}

        prefix_candidate_batch = self.tokenizer(sentences, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, add_prefix_space=True)
        prefix_candidate_batch = {k:v.to(self.device) for k, v in prefix_candidate_batch.items()}

        encoder = self.model.get_encoder()
        encoder_states = encoder(**src_batch).last_hidden_state
        encoder_states = encoder_states.expand(k, *encoder_states.shape[1:])        

        # key: sentence ids separated with comma
        # value: information including {prefix_token_id, cache, best_score, best_length}
        beam2info = {'': {
            'prefix_token_id': self.model.config.decoder_start_token_id, # last token
            'raw_sentence_score': 0.0,
            'extended_length': 0,
            "beam_score": 0,
            'cache': None,
        }}  

        for step in range(max_length):
            if step == 0:
                labels = candidate_batch["input_ids"]
                mask = candidate_batch["attention_mask"]
            else:
                labels = prefix_candidate_batch["input_ids"]
                mask = prefix_candidate_batch["attention_mask"]

            lengths = mask.sum(-1)

            keys = list(beam2info.keys())
            for beam in keys:
                info = beam2info[beam]
                selected = [int(i) for i in beam.split(',') if i]
                if len(selected) != step:
                    continue

                extended_lengths = (info['extended_length'] + lengths).long()
                
                decoder_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id, info['prefix_token_id'])

                cache = info['cache']() if info['cache'] else None  # compute lazy evaluation
                step_output = self.model(
                    encoder_outputs=(encoder_states,),
                    decoder_input_ids=decoder_ids,
                    past_key_values=cache,
                    use_cache=True)

                logprobs = torch.log_softmax(step_output.logits, -1)
                token_scores = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
                raw_sentence_scores = (token_scores * mask).sum(-1)
                raw_sentence_scores = info['raw_sentence_score'] + raw_sentence_scores

                # update states
                cache = step_output["past_key_values"]  # batch_size * ...
                
                for i in range(len(raw_sentence_scores)):
                    if i in selected:
                        continue
                        
                    new_beam = ','.join([str(j) for j in selected + [i]])
                    new_beam_cache = _lazy_reorder_cache(cache, i, extended_lengths[i])
                    # new_beam_cache = _reorder_cache(cache, i, extended_lengths[i])
                    new_beam_prefix_token_id = labels[i, lengths[i]-1]

                    beam2info[new_beam] = {
                        'cache': new_beam_cache,
                        'prefix_token_id': new_beam_prefix_token_id,
                        'raw_sentence_score': raw_sentence_scores[i],
                        'extended_length': extended_lengths[i],
                        'beam_score': raw_sentence_scores[i] / (extended_lengths[i] ** alpha)
                    }

                del beam2info[beam]
                
            # prune
            if len(beam2info) > beam_size:
                new_beam2info = {}
                for k, v in sorted(beam2info.items(), key=lambda item: item[1]['beam_score'], reverse=True)[:beam_size]:
                    new_beam2info[k] = v

                beam2info = new_beam2info
        
        beam_selections = [[int(i) for i in beam.split(',')] for beam in beam2info.keys()]
        if not beam_selections:
            beam_selections = [[]]
        beam_summaries = ["\n".join(sentences[i] for i in indices) for indices in beam_selections]
            
        return {
            "selected": beam_selections,
            "summary": beam_summaries,
            "scores": [v["beam_score"].item() for v in  beam2info.values()]
        }

def _get_trigrams(s):
    return set(nltk.trigrams(nltk.word_tokenize(s)))

class BeamSearchExtended(ModelBase):
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(
        self,
        text: str,
        sentences: List[str],
        alpha: int = 1,
        min_length: int = 0,
        max_length: int = 3,
        beam_size: int = 1,
        early_stop: bool = False,
        monotonic: bool = False,
        clip: float = None,  # allows for clipping log probabilities e.g. (-.5, None)
        block_permutation_size: int = 0,
        src_max_length=512,
        trigram_block: bool = False) -> dict:

        # hack
        selected_trigrams = set()
        sentence_trigrams = None
        if trigram_block:
            assert beam_size == 1, "trigram block not supported for beam search"
            sentence_trigrams = [_get_trigrams(s.lower()) for s in sentences]

        acc = []
        if early_stop:
            sentences = sentences + [self.tokenizer.eos_token]

        k = len(sentences)
        eos_index = k - 1

        src_batch = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=src_max_length)
        src_batch = {k: v.to(self.device) for k, v in src_batch.items()}        

        bos_sentences = [self.tokenizer.bos_token+s for s in sentences]
        candidate_batch = self.tokenizer(bos_sentences, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)
        candidate_batch = {k:v.to(self.device) for k, v in candidate_batch.items()}

        prefix_candidate_batch = self.tokenizer(sentences, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False, add_prefix_space=True)
        prefix_candidate_batch = {k:v.to(self.device) for k, v in prefix_candidate_batch.items()}

        encoder = self.model.get_encoder()
        encoder_states = encoder(**src_batch).last_hidden_state
        encoder_states = encoder_states.expand(k, *encoder_states.shape[1:])        

        # key: sentence ids separated with comma
        # value: information including {prefix_token_id, cache, best_score, best_length}
        beam2info = {'': {
            'prefix_token_id': self.model.config.decoder_start_token_id, # last token
            'raw_sentence_score': 0.0,
            'extended_length': 0,
            "beam_score": 0,
            'cache': None,
        }}  

        for step in range(max_length):
            if step == 0:
                labels = candidate_batch["input_ids"]
                mask = candidate_batch["attention_mask"]
            else:
                labels = prefix_candidate_batch["input_ids"]
                mask = prefix_candidate_batch["attention_mask"]

            lengths = mask.sum(-1)

            keys = list(beam2info.keys())
            
            permutation_set = [frozenset([int(i) for i in beam.split(',') if i]) for beam in keys]
            for beam in keys:
                info = beam2info[beam]
                selected = [int(i) for i in beam.split(',') if i]
                if trigram_block:
                    selected_trigrams = set()
                    for i in beam.split(','):
                        if not i:
                            continue
                        selected_trigrams.update(sentence_trigrams[int(i)])

                if early_stop and selected and selected[-1] == eos_index:  # keep + dont expand beam
                    # forward the beam if EOS has been encountered
                    continue

                extended_lengths = (info['extended_length'] + lengths).long()
                
                decoder_ids = shift_tokens_right(labels, self.tokenizer.pad_token_id, info['prefix_token_id'])

                cache = info['cache']() if info['cache'] else None  # lazy evaluation to reduce mem
                step_output = self.model(
                    encoder_outputs=(encoder_states,),
                    decoder_input_ids=decoder_ids,
                    past_key_values=cache,
                    use_cache=True)

                logprobs = torch.log_softmax(step_output.logits, -1)
                token_scores = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
                masked_token_scores = (token_scores * mask)

                if clip:
                    masked_token_scores = torch.clamp(masked_token_scores, min=clip[0], max=clip[1])

                raw_sentence_scores = masked_token_scores.sum(-1)
                raw_sentence_scores = info['raw_sentence_score'] + raw_sentence_scores

                if len(selected) <= min_length:
                    raw_sentence_scores[eos_index] = -float("inf")

                # update states
                cache = step_output["past_key_values"]  # batch_size * ...
                
                for i in range(len(raw_sentence_scores)):
                    if monotonic and selected and i <= max(selected):
                        continue

                    if i in selected:
                        continue

                    if trigram_block and selected_trigrams.intersection(sentence_trigrams[i]):
                        continue
                    
                    # block "ngrams" of size `block_permutation_size`
                    if block_permutation_size > 0:
                        q_gram = frozenset(selected[:block_permutation_size-1] + [i])
                        if len(q_gram) == block_permutation_size and q_gram in permutation_set:
                            continue
                        
                    new_beam = ','.join([str(j) for j in selected + [i]])
                    new_beam_cache = _lazy_reorder_cache(cache, i, extended_lengths[i])
                    # new_beam_cache = _reorder_cache(cache, i, extended_lengths[i])
                    new_beam_prefix_token_id = labels[i, lengths[i]-1]

                    beam2info[new_beam] = {
                        'cache': new_beam_cache,
                        'prefix_token_id': new_beam_prefix_token_id,
                        'raw_sentence_score': raw_sentence_scores[i],
                        'extended_length': extended_lengths[i],
                        'beam_score': raw_sentence_scores[i] / (extended_lengths[i] ** alpha)
                    }
                # patch to not give empty summary if there are no beams in addition to the original
                if len(beam2info) == 1:
                    print("CAREFUL! Only 1 beam available for search??")
                    break
                else:
                    del beam2info[beam]  # beams aren't removed if they end with EOS. skipped above                
                
            # prune beams with the top scoring values
            if len(beam2info) > beam_size:
                new_beam2info = {}
                for key, value in sorted(beam2info.items(), key=lambda item: item[1]['beam_score'], reverse=True)[:beam_size]:
                    new_beam2info[key] = value

                beam2info = new_beam2info

        beam_selections = [[int(i) for i in beam.split(',')] for beam in beam2info.keys()]
        if not beam_selections:  # weird edge case where there is empty sentences??
            beam_selections = [[]]

        if early_stop:
            # postprocess to truncate EOS candidate for predictions
            beam_selections = [[i if i < eos_index else -1 for i in s] for s in beam_selections]
        
        beam_summaries = ["\n".join(sentences[i] for i in indices if i >= 0) for indices in beam_selections]
 
        return {
            "selected": beam_selections,
            "summary": beam_summaries,
            "scores": [v["beam_score"].item() for v in  beam2info.values()]
        }


class RankModel(ModelBase):
    def __init__(self, tokenizer: BartTokenizer, model: BartForConditionalGeneration) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

    @torch.no_grad()
    def __call__(self, text: str, candidates: list[str], alpha: int = 1, max_src_length=512) -> dict:
        k = len(candidates)
        
        src_batch = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=max_src_length)
        src_batch = {k: v.to(self.device) for k, v in src_batch.items()}

        encoder = self.model.get_encoder()
        encoder_states = encoder(**src_batch).last_hidden_state
        encoder_states = encoder_states.expand(k, *encoder_states.shape[1:])

        candidate_batch = self.tokenizer(candidates, truncation=True, padding=True, return_tensors="pt")
        candidate_batch = {k: v.to(self.device) for k, v in candidate_batch.items()}

        candidate_ids = candidate_batch["input_ids"]
        candidate_lengths = candidate_batch["attention_mask"].sum(-1)

        prefix_token_id = self.model.config.decoder_start_token_id
        decoder_ids = shift_tokens_right(candidate_ids, self.tokenizer.pad_token_id, prefix_token_id)

        step_output = self.model(
            encoder_outputs=(encoder_states,),
            decoder_input_ids=decoder_ids)

        logprobs = torch.log_softmax(step_output.logits, -1)
        token_scores = torch.gather(logprobs, -1, candidate_ids.unsqueeze(-1)).squeeze()
        raw_sentence_scores = (token_scores * candidate_batch["attention_mask"]).sum(-1)
        sentence_scores = raw_sentence_scores / (candidate_lengths ** alpha)

        best_idx = sentence_scores.argmax()

        return {
            "summary": candidates[best_idx],
            "probabilities": sentence_scores,
        }