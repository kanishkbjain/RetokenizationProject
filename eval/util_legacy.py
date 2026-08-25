import os
import torch
import contextlib

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, AutoConfig, LogitsProcessor, LogitsProcessorList
from huggingface_hub import HfApi
import numpy as np
from pathlib import Path
# from olmo.util import ensure_dir
import json
import pandas as pd
from methodtools import lru_cache
import random
from string import ascii_uppercase

from transformers import logging
import pickle
import gc
from torch.nn import functional as F
# Set the verbosity to INFO
logging.set_verbosity_error()



class EntropyAndLogprobTracker(LogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

        # One-step lag logits (unscaled) on CPU
        self.prev_scores_unscaled = None

        # Per-step entropies (T_steps, B_total)
        self.unnorm_entropies = []  # unscaled entropy
        self.norm_entropies = []    # scaled entropy

        # Per-sequence per-token logprobs (lists of lists)
        self.answer_logprobs_scaled = None     # list[list[float]]
        self.answer_logprobs_unscaled = None   # list[list[float]]
        self.answer_logits_unscaled = None   # list[list[float]]

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:

        oncpu=False
        # scores: (B_total, vocab) on device
        # print(input_ids.shape[-1], end='\r')
        scores_cpu = scores.detach().float().cpu() if oncpu else scores
        B_total, V = scores_cpu.shape

        # ---- 1) Entropies for *current* scores ----
        dist_unscaled = torch.distributions.Categorical(logits=scores_cpu)
        H_unscaled = dist_unscaled.entropy()  # (B_total,)

        H_unscaled = H_unscaled.detach().cpu() if oncpu else H_unscaled
        self.unnorm_entropies.append(H_unscaled)

        if self.temperature != 1.0:
            dist_scaled = torch.distributions.Categorical(
                logits=scores_cpu / self.temperature
            )
            H_scaled = dist_scaled.entropy()
        else:
            H_scaled = H_unscaled
        H_scaled = H_scaled.detach().cpu() if oncpu else H_scaled
        self.norm_entropies.append(H_scaled)


        # ---- 2) Logprobs for the token chosen at previous step ----
        # At step t, input_ids' last token is the one sampled at t-1
        if self.prev_scores_unscaled is not None:
            # lazy init nested lists
            if self.answer_logprobs_scaled is None:
                self.answer_logprobs_scaled = [[] for _ in range(B_total)]
                self.answer_logprobs_unscaled = [[] for _ in range(B_total)]
                self.answer_logits_unscaled = [[] for _ in range(B_total)]

            prev_scores = self.prev_scores_unscaled             # (B_total, vocab)
            prev_logprobs_unscaled = F.log_softmax(prev_scores, dim=-1)
            prev_logprobs_scaled = F.log_softmax(
                prev_scores / self.temperature, dim=-1
            )

            last_tokens = input_ids[:, -1].cpu()                # token at t-1
            idx = torch.arange(B_total)

            lp_unscaled = prev_logprobs_unscaled[idx, last_tokens]  # (B_total,)
            lp_scaled = prev_logprobs_scaled[idx, last_tokens]      # (B_total,)
            logits_unscaled = prev_scores[idx, last_tokens]      # (B_total,)

            lp_scaled = lp_scaled.detach().cpu() if oncpu else lp_scaled
            lp_unscaled = lp_unscaled.detach().cpu() if oncpu else lp_unscaled
            logits_unscaled = logits_unscaled.detach().cpu() if oncpu else logits_unscaled

            for b in range(B_total):
                self.answer_logprobs_unscaled[b].append(float(lp_unscaled[b]))
                self.answer_logprobs_scaled[b].append(float(lp_scaled[b]))
                self.answer_logits_unscaled[b].append(float(logits_unscaled[b]))

        # ---- 3) Update rolling logits ----
        self.prev_scores_unscaled = scores_cpu if oncpu else scores.detach().cpu()

        # Return scores unchanged so generation proceeds as usual
        return scores

    def finalize(self, sequences: torch.LongTensor,):

        prev_scores = self.prev_scores_unscaled  # (B_total, vocab)

        prev_logprobs_unscaled = F.log_softmax(prev_scores, dim=-1)
        prev_logprobs_scaled = F.log_softmax(
            prev_scores / self.temperature, dim=-1
        )

        last_tokens = sequences[:, -1].cpu()  # token at t-1
        B_total  = sequences.shape[0]
        idx = torch.arange(B_total)

        lp_unscaled = prev_logprobs_unscaled[idx, last_tokens]  # (B_total,)
        lp_scaled = prev_logprobs_scaled[idx, last_tokens]  # (B_total,)
        logits_unscaled = prev_scores[idx, last_tokens]  # (B_total,)

        for b in range(B_total):
            self.answer_logprobs_unscaled[b].append(float(lp_unscaled[b]))
            self.answer_logprobs_scaled[b].append(float(lp_scaled[b]))
            self.answer_logits_unscaled[b].append(float(logits_unscaled[b]))



        entropy_unnorm = torch.stack(self.unnorm_entropies, dim=0).T  # (B_total, T_steps)
        entropy_norm = torch.stack(self.norm_entropies, dim=0).T


        answer_logprobs_scaled = torch.tensor(self.answer_logprobs_scaled) # (B_total, T_steps)
        answer_logprobs_unscaled = torch.tensor(self.answer_logprobs_unscaled) # (B_total, T_steps)
        answer_logits_unscaled = torch.tensor(self.answer_logits_unscaled) # (B_total, T_steps)


        return entropy_unnorm, entropy_norm, answer_logprobs_scaled, answer_logprobs_unscaled, answer_logits_unscaled


class generateSegments:

    def __init__(self, vocabulary):
        self.vocab = vocabulary

    @lru_cache()
    def countSegments(self, word, start):
        wordlen = len(word)
        if start == wordlen:
            return 1
        total = 0
        for end in range(start + 1, wordlen + 1):
            if word[start:end] in self.vocab:
                total += self.countSegments(word, end)
        return total

    def buildSegments(self, word, start):
        wordlen = len(word)
        if start == wordlen:
            return []
        choices = []
        weights = []
        for end in range(start + 1, wordlen + 1):
            segment = word[start:end]
            if segment in self.vocab:
                count = self.countSegments(word, end)
                if count > 0:
                    choices.append(segment)
                    weights.append(count)
        if not choices:
            return []
        nextSegment = random.choices(choices, weights=weights, k=1)[0]
        return [nextSegment] + self.buildSegments(word, start + len(nextSegment))

    def generate(self, word):
        total = self.countSegments(word, 0)
        if total == 0:
            return [] # If there is no valid way to segment word
        return self.buildSegments(word, 0)

    @lru_cache()
    def _all_segments_cached(self, word, start):
        """
        Returns all segmentations from `start` to end as tuples of tuples for caching.
        Each result is a tuple representing one segmentation path.
        """
        wordlen = len(word)
        if start == wordlen:
            return ((),)  # one valid path: empty continuation

        results = []
        for end in range(start + 1, wordlen + 1):
            seg = word[start:end]
            if seg in self.vocab:
                tails = self._all_segments_cached(word, end)
                for tail in tails:
                    results.append((seg,) + tail)

        return tuple(results)

    def get_all_segments(self, word):
        """
        Returns all valid segmentations of `word` as a list of lists.
        Example: for word='abcd' and vocab={'a','ab','b','cd','c','d'},
        possible outputs: [['ab','cd'], ['a','b','cd'], ['a','bc','d'], ...]
        """
        paths = self._all_segments_cached(word, 0)
        return [list(path) for path in paths]

    def get_first_tokens(self, word):
        """
        Returns the set of first tokens for all valid segmentations of `word`.
        Only includes a first token if at least one complete segmentation exists.
        Example: for word='Ġlivestock' with vocab allowing
        ['Ġ','live','stock'] and ['Ġl','i','v','e','s','t','o','c','k'],
        returns ['Ġ','Ġl'] (order not guaranteed).
        """
        first_tokens = []
        seen = set()
        wordlen = len(word)
        for end in range(1, wordlen + 1):
            seg = word[:end]
            if seg in self.vocab:
                # Check if continuation from `end` can complete
                if self.countSegments(word, end) > 0 and seg not in seen:
                    seen.add(seg)
                    first_tokens.append(seg)
        return first_tokens



def prep_incontext_examples(test_df, num_incontext_examples):
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }
    return incontext_indices


def parse_number(output_str, output_type="int"):
    output_str = output_str.strip().replace(",", "")
    output_num = None
    try:
        if output_type == "int":
            output_num = int(output_str)
        elif output_type == "float":
            output_num = float(output_str)
    except ValueError:
        print(f"Failed to parse number: {output_str}")
        pass
    return output_num


def format_example(
    question, passage=None, choices=None, answer=None, qa_format="qnan", question_prefix="Question:", answer_prefix= "Answer:"
):
    """Options for QA format:
    qa: Question: {question}\nAnswer: {answer}
    qnan: Question:\n{question}\nAnswer:\n{answer}
    qna: Question:\n{question}\nAnswer: {answer}
    q: Question: {question} (if answer=None, else equivalent to qa)
    """
    text = ""
    if passage:
        text += f"{passage.strip()}\n\n"

    text += question_prefix + "\n" if "qn" in qa_format else question_prefix + " "
    text += question.strip() + "\n"

    if choices:
        for label, choice in zip(ascii_uppercase, choices):
            text += f"{label}. {choice.strip()}\n"

    if answer or qa_format != "q":
        text += answer_prefix + "\n" if "an" in qa_format else answer_prefix
    if answer:
        if isinstance(answer, str):
            answer = answer.strip()
        answer = str(answer)
        text += answer if "an" in qa_format else " " + answer

    return text


def parse_mc_pred(output, num_options=4, qa_format="qnan"):
    """
    Parses the predicted MC option (e.g., "A") from the model output.
    Returns None if the output is not a valid MC option.
    """
    parsed_answer = None
    valid = True
    if qa_format == "q":
        if output.startswith("Answer:"):  # output answer should start with "Answer: "
            output = output.replace("Answer: ", "")
        else:
            valid = False
    elif qa_format in ["qa", "qna"]:
        if output.startswith(" "):  # output answer should start with leading space
            output = output.lstrip()
        else:
            valid = False

    if output and valid and (output[0] in ascii_uppercase[:num_options]):
        parsed_answer = output[0]

    return parsed_answer


def get_checkpoints(model_name):
    refs = HfApi().list_repo_refs(model_name)
    checkpoints = []
    for branch in refs.branches:
        checkpoints.append(branch.name)
    return checkpoints


def fiddle_tokens(inputTensor, attention_mask, tokenizer, segmentor, is_mcq, reason, retokenizationp, no_prefix_suffix=False):
    inputList = inputTensor.tolist()
    fiddledPrompts = []

    if no_prefix_suffix:
        input_ids_prefix = []
        input_ids_suffix = []
    else:
        if is_mcq:
            if reason:
                input_ids_prefix = tokenizer.encode(
                    """<|im_start|>system\n Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering. <|im_end|>\n<|im_start|>user\n""", add_special_tokens=False)
                # input_ids_prefix = tokenizer.encode(
                #     """<|im_start|>system\nYou will be given a multiple-choice question. Answer the question in 3-4 sentences, ending it by placing the final answer choice as a single letter in square brackets (e.g., [A]). <|im_end|>\n<|im_start|>user\n""",
                #     add_special_tokens=False
                # )
            # input_ids_prefix = tokenizer.encode(
            #     """<|im_start|>system\nYou are a helpful assistant. For the following multiple choice question with four answer choices, provide a short reasoning for what should be the right answer, then place only your chosen answer option within braces {} (like for example {E}).<|im_end|>\n<|im_start|>user\n""",
            #     add_special_tokens=False
            # )
            else:
                input_ids_prefix = tokenizer.encode(
                    """<|im_start|>system\nYou are a helpful assistant. For the following multiple choice questions, return the answer only, without any additional reasoning or explanation. <|im_end|>\n<|im_start|>user\n""",
                    add_special_tokens=False)
        elif not no_prefix_suffix:
            input_ids_prefix = tokenizer.encode(
                """<|im_start|>system\nYou are a helpful assistant. <|im_end|>\n<|im_start|>user\n""", #For the following question, return the answer only, without any additional reasoning or explanation.
                add_special_tokens=False
            )
        input_ids_suffix = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False
        )

    lengths = []
    for i, prompt in enumerate(inputList):
        fiddledPrompt = []
        realTokens = []
        tokenPrompt = tokenizer.convert_ids_to_tokens(prompt)
        for j, token in enumerate(tokenPrompt):
            if (attention_mask[i][j] == 1): # non-filler (real) tokens
                realTokens.append(token)
        for token in realTokens:
            if token == tokenizer.bos_token:
                continue
            if retokenizationp == 0.0 or (random.random() >= retokenizationp):
                fiddledPrompt.append(token)
            else:
                fiddledPrompt += segmentor.generate(token)
        lengths.append((len(input_ids_prefix), len(fiddledPrompt), len(input_ids_suffix)))
        fiddledPrompts.append(input_ids_prefix + tokenizer.convert_tokens_to_ids(fiddledPrompt) + input_ids_suffix)

    return tokenizer.pad({"input_ids": fiddledPrompts}, padding="longest", padding_side="left", return_tensors="pt"), lengths


def batched_generate(output_dir, prompts, model, tokenizer, batch_size=5, is_mcq=False, retokenizationp=1.0,
                     temperature=1.0, num_sequences=10, top_k=50, reason=True, **generation_kwargs):

    print('Generating with reasoning set to', reason)
    if reason:
        output_dir = os.path.join(output_dir, "reasoning")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(output_dir, "no_reasoning")
        os.makedirs(output_dir, exist_ok=True)

    outpath = os.path.join(output_dir, f"prompt%i_pval{retokenizationp:0.02f}_"
                                       f"seqs{num_sequences}_topk{top_k}_"
                                       f"temp{temperature}.jsonl")
    prompts_total_number = len(prompts)
    prompts_i = list(range(len(prompts)))
    for i in range(len(prompts)):
        if os.path.exists(outpath % i):
            print(f"Prompt {i} already generated, skipping.", end='\r')
            prompts_i.remove(i)
    print(f"Prompts to generate: {len(prompts_i)}, total: {prompts_total_number}")
    prompts = [prompts[i] for i in prompts_i]

    print(len(prompts), "prompts to generate.")
    print(prompts_i[:10], "(and more) prompt indices to generate.")
    if isinstance(tokenizer, GPT2Tokenizer):
        segmentor = generateSegments(tokenizer.get_vocab().keys())
    else:
        segmentor = generateSegments(tokenizer.vocab.keys())

    pbar = tqdm(total=len(prompts), desc=f"Temp : {temperature} P : {retokenizationp}")

    use_autocast = (model.device.type == "cuda")

    # get available gpu memory fro all gpus


    # available_gpu_mem = 0
    # for i in range(torch.cuda.device_count()):
    #     available_gpu_mem += torch.cuda.mem_get_info(i)[0]
    #
    # print(f"Initial available GPU memory: {available_gpu_mem/(1024*1024*1024):0.2f} GB")
    skipped_count = 0
    skipped_tolerance = min(25, len(prompts)//10)  # allow up to 10% skips, max 25
    for i in range(0, len(prompts), batch_size):

        #if available gpu memory decreases a lot during generation, print a warning
        # available_gpu_mem_ = 0
        # for gi in range(torch.cuda.device_count()):
        #     available_gpu_mem_ += torch.cuda.mem_get_info(gi)[0]

        # if (available_gpu_mem - available_gpu_mem_) / available_gpu_mem > 0.1:
        #     print(f"Warning: Available GPU memory decreased significantly during generation to {available_gpu_mem_/(1024*1024*1024):0.2f} GB")
        # elif (available_gpu_mem - available_gpu_mem_) / available_gpu_mem < -0.1:
        #     print(f"Info: Available GPU memory increased significantly during generation to {available_gpu_mem_/(1024*1024*1024):0.2f} GB")

        entropy_tracker = EntropyAndLogprobTracker(temperature=temperature)
        logits_processors = LogitsProcessorList([entropy_tracker])

        batch_prompts = prompts[i: i + batch_size]
        batch_prompts_i = prompts_i[i: i + batch_size]

        # Tokenize on CPU
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
        )

        # Still CPU here
        res, batch_token_lengths = fiddle_tokens(
            batch_inputs.input_ids,
            batch_inputs.attention_mask,
            tokenizer,
            segmentor,
            is_mcq,
            reason,
            retokenizationp,
        )
        # print(tokenizer.batch_decode(res["input_ids"][:2], skip_special_tokens=False))

        # Move only needed tensors to GPU
        res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in res.items()}

        # -------- GENERATION + ANSWER ENTROPIES --------
        with torch.inference_mode():
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                try:
                    batch_outputs_answer = model.generate(
                        **res,
                        num_return_sequences=num_sequences,
                        do_sample=True,
                        num_beams=1,
                        top_k=top_k,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id,
                        tokenizer=tokenizer,
                        output_scores=False,
                        output_logits=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        temperature=temperature,
                        max_length=10000,
                        eos_token_id=tokenizer.eos_token_id,
                        logits_processor=logits_processors,
                        **generation_kwargs,
                    )
                except RuntimeError as e:
                    # print(f"RuntimeError during generation: {e}")
                    # print("Skipping this batch.")
                    # pbar.update(len(batch_prompts))
                    # print(f"Skipping {i}:{i+batch_size}", end='\r')
                    print(f"Skipping {outpath% batch_prompts_i[0]} due to OOM")
                    skipped_count += len(batch_prompts)
                    del res
                    gc.collect()
                    torch.cuda.empty_cache()
                    if skipped_count > skipped_tolerance:
                        raise e
                    else:
                        continue
        # Immediately move sequences off GPU (huge tensor)
        sequences = batch_outputs_answer.sequences.detach().cpu()
        c_len = res["input_ids"].shape[1]  # prompt length (after padding)

        entropy_unnorm_answer, entropy_norm_answer, \
            answer_logprobs_scaled, answer_logprobs_unscaled,\
            answer_logits_unscaled = entropy_tracker.finalize(
            sequences,)


        del batch_inputs, batch_outputs_answer, entropy_tracker, logits_processors
        gc.collect()
        torch.cuda.empty_cache()

        # -------- PROMPT ENTROPIES --------
        with torch.inference_mode():
            with (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                  if use_autocast else contextlib.nullcontext()):
                logits_prompt = model(**res, return_dict=True,).logits

        # Prompt logits -> CPU ASAP
        unscaled_logits_prompt = logits_prompt.detach().cpu()

        del logits_prompt
        gc.collect()
        torch.cuda.empty_cache()

        scaled_logits_prompt = (unscaled_logits_prompt / temperature)
        entropy_unnorm_prompt = torch.distributions.Categorical(
            logits=unscaled_logits_prompt
        ).entropy()
        entropy_norm_prompt = torch.distributions.Categorical(
            logits=scaled_logits_prompt
        ).entropy()

        # `res` -> CPU for saving / logging
        res_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in res.items()}

        # -------- WRITE RESULTS (only CPU tensors) --------
        write_batch_results(
            outpath,
            batch_prompts,
            batch_prompts_i,
            num_sequences,
            sequences,  # sequences are on CPU; scores/logits removed
            batch_token_lengths,
            res_cpu,
            scaled_logits_prompt,
            unscaled_logits_prompt,
            entropy_norm_prompt,
            entropy_unnorm_prompt,
            answer_logprobs_scaled,
            answer_logprobs_unscaled,
            answer_logits_unscaled,
            entropy_norm_answer,
            entropy_unnorm_answer,
        )

        pbar.update(len(batch_prompts))

        # Extra safety: drop references, collect garbage
        del (
            res,
            res_cpu,
            answer_logprobs_scaled,
            answer_logprobs_unscaled,
            answer_logits_unscaled,
            entropy_unnorm_answer,
            entropy_norm_answer,
            unscaled_logits_prompt,
            scaled_logits_prompt,
            entropy_unnorm_prompt,
            entropy_norm_prompt,
        )

        gc.collect()
        torch.cuda.empty_cache()

    # Print total items and how many were skipped due to errors
    done_ = sum([1 for i in range(prompts_total_number) if os.path.exists(outpath % i)])
    print(f"Generation completed. Total prompts: {prompts_total_number}, Completed: {done_}, Skipped: {prompts_total_number - done_}")
    return 0


import torch
import torch.nn.functional as F
import json


def write_batch_results(
    outpath,
    batch_prompts,
    batch_prompts_i,
    num_sequences,
    sequences,
    batch_token_lengths,
    res,
    scaled_logits_prompt,
    unscaled_logits_prompt,
    entropy_norm_prompt,
    entropy_unnorm_prompt,
    answer_logprobs_scaled,
    answer_logprobs_unscaled,
    answer_logits_unscaled,
    entropy_norm_answer,
    entropy_unnorm_answer,
):
    batch_size = len(batch_prompts)

    # --- Sanity checks: everything should be on CPU now ---
    assert scaled_logits_prompt.device.type == "cpu"
    assert unscaled_logits_prompt.device.type == "cpu"
    assert answer_logprobs_scaled.device.type == "cpu"
    assert answer_logprobs_unscaled.device.type == "cpu"
    assert sequences.device.type == "cpu"
    assert res["input_ids"].device.type == "cpu"

    # Prompt length in tokens
    c_len = res["input_ids"].shape[1]
    # ----------------------------------------------------
    # PROMPT-LEVEL STUFF
    # ----------------------------------------------------
    # Use attention mask if present, otherwise just take all tokens.
    attn = res.get("attention_mask")
    if attn is None:
        prompt_generated_ids = [seq.tolist() for seq in res["input_ids"]]
    else:
        prompt_generated_ids = [
            [tok.item() for tok, m in zip(seq, mask) if m.item() == 1]
            for seq, mask in zip(res["input_ids"], attn)
        ]

    # Compute log probs once
    prompt_logprobs_logits = F.log_softmax(scaled_logits_prompt, dim=-1)
    # Add unscaled prompt logprobs
    prompt_logprobs_unscaled_logits = F.log_softmax(unscaled_logits_prompt, dim=-1)

    prompt_logprobs = [
        prompt_logprobs_logits[ti].gather(
            -1, res["input_ids"][ti].unsqueeze(-1)
        ).view(-1).tolist()
        for ti in range(batch_size)
    ]
    # Gather unscaled prompt logprobs per token
    prompt_logprobs_unscaled = [
        prompt_logprobs_unscaled_logits[ti].gather(
            -1, res["input_ids"][ti].unsqueeze(-1)
        ).view(-1).tolist()
        for ti in range(batch_size)
    ]

    prompt_unscaled_logits = [
        unscaled_logits_prompt[ti].gather(
            -1, res["input_ids"][ti].unsqueeze(-1)
        ).view(-1).tolist()
        for ti in range(batch_size)
    ]

    # Entropies are already CPU; slice to c_len for safety (though should match)
    prompt_scaled_xentropy = entropy_norm_prompt[:, :c_len].tolist()
    prompt_unscaled_xentropy = entropy_unnorm_prompt[:, :c_len].tolist()

    # ----------------------------------------------------
    # ANSWER-LEVEL STUFF
    # ----------------------------------------------------
    # scaled_logits_answer: (T_answer, batch_size * num_sequences, vocab)
    # Precompute log-softmax once


    answer_generated_ids = []
    answer_generated_logprobs = []
    answer_generated_logprobs_unscaled = []
    answer_generated_unscaled_logits = []
    answer_generated_scaled_xentropy = []
    answer_generated_unscaled_xentropy = []

    for batch_i in range(batch_size):
        # indices corresponding to this prompt's sequences in the flattened dimension
        start = batch_i * num_sequences
        end = (batch_i + 1) * num_sequences

        # Entropy slices for this prompt (shape: T_answer x num_sequences)
        answer_generated_scaled_xentropy.append(
            entropy_norm_answer[start:end].tolist()
        )
        answer_generated_unscaled_xentropy.append(
            entropy_unnorm_answer[start:end].tolist()
        )

        # Per-sequence info
        batch_answer_ids = []
        batch_answer_logprobs = []
        # NEW: per-sequence unscaled logprobs
        batch_answer_logprobs_unscaled = []
        batch_answer_unscaled_logits = []

        for seqi in range(num_sequences):
            seqidx = start + seqi

            # sequence tokens after the prompt (strip prompt length c_len)
            seq = sequences[seqidx, c_len:]

            # IDs
            batch_answer_ids.append(seq.tolist())

            # Logprobs and logits gathered from time dimension for this seq
            # answer_logprobs_logits: (T_answer, B_seq, vocab)
            # -> (T_answer, answer_len)
            seq_logprobs = answer_logprobs_scaled[seqidx]
            seq_logprobs_unscaled = answer_logprobs_unscaled[seqidx]
            seq_unscaled_logits = answer_logits_unscaled[seqidx]

            batch_answer_logprobs.append(seq_logprobs.tolist())
            batch_answer_logprobs_unscaled.append(seq_logprobs_unscaled.tolist())
            batch_answer_unscaled_logits.append(seq_unscaled_logits.tolist())

        answer_generated_ids.append(batch_answer_ids)
        answer_generated_logprobs.append(batch_answer_logprobs)
        answer_generated_logprobs_unscaled.append(batch_answer_logprobs_unscaled)
        answer_generated_unscaled_logits.append(batch_answer_unscaled_logits)

    # Pre / prompt / post lengths
    pre_prompt_post_lens = list(batch_token_lengths)

    # ----------------------------------------------------
    # WRITE OUT JSON (one file per prompt)
    # ----------------------------------------------------
    for batch_i in range(batch_size):
        result = {
            "prompt": batch_prompts[batch_i],
            "prompt_generated_ids": prompt_generated_ids[batch_i],
            "prompt_logprobs": prompt_logprobs[batch_i],
            "prompt_logprobs_unscaled": prompt_logprobs_unscaled[batch_i],
            "prompt_unscaled_logits": prompt_unscaled_logits[batch_i],
            "prompt_scaled_xentropy": prompt_scaled_xentropy[batch_i],
            "prompt_unscaled_xentropy": prompt_unscaled_xentropy[batch_i],
            "answer_generated_ids": answer_generated_ids[batch_i],
            "answer_generated_logprobs": answer_generated_logprobs[batch_i],
            "answer_generated_logprobs_unscaled": answer_generated_logprobs_unscaled[batch_i],
            "answer_generated_unscaled_logits": answer_generated_unscaled_logits[batch_i],
            "answer_scaled_xentropy": answer_generated_scaled_xentropy[batch_i],
            "answer_unscaled_xentropy": answer_generated_unscaled_xentropy[batch_i],
            "pre_prompt_post_lens": pre_prompt_post_lens[batch_i],
        }

        with open(outpath % batch_prompts_i[batch_i], "w") as fo:
            json.dump(result, fo)

    return

def load_tokenizer(tokenizer_name_or_path, padding_side="left"):
    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.backend_tokenizer.model.dropout = 0.0  # always use dropout p = 0.0 for inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer

def load_model(model_name_or_path, step=None):
    revision = None
    if os.path.exists(model_name_or_path):
        if step:
            model_name_or_path += f"/step{step}"
    else:
        if step:
            try:
                revision = [r for r in get_checkpoints(model_name_or_path) if r.split("-")[1] == f"step{step}"][0]
                print(f"Revision: {revision}")
            except IndexError:
                raise ValueError(f"Checkpoint {step} not found")

    print(f"Loading model from {model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        revision=revision if "allenai" in model_name_or_path else None,
        # force_download=True
    )
    model.eval()

    return model

def load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path=None, step=None, padding_side="left"):
    model = load_model(model_name_or_path, step=step)
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = load_tokenizer(tokenizer_name_or_path, padding_side=padding_side)

    return model, tokenizer


def write_results(results, output_dir, metric="accuracy", print_metrics=False):
    metrics = {"num_examples": len(results), "accuracy": np.mean([r["correct"] for r in results])}

    if "valid" in results[0]:
        metrics["valid_answer"] = np.mean([r["valid"] for r in results])

    if "split" in results[0]:
        for split in sorted(set([r["split"] for r in results])):
            split_results = [r for r in results if r["split"] == split]
            metrics[f"{split}_accuracy"] = np.mean([r["correct"] for r in split_results])

    if print_metrics:
        for k, v in metrics.items():
            print(f"{k}: {v}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    with open(output_dir / "metrics.json", "w") as fo:
        json.dump(metrics, fo, indent=4)
    with open(output_dir / "example_prompt.txt", "w") as fo:
        fo.write(results[0]["prompt"])
    pd.DataFrame(results).to_json(output_dir / "predictions.jsonl", orient="records", lines=True)



PRECISION_MAP = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,  # theoretical; most runtimes still use fp16/bf16 for KV
}

def calculate_kv_cache_size_per_token(model, precision: str = "fp16") -> float:
    """
    Returns KV cache size in MB **per generated token per sequence per batch**.
    Uses your existing estimate_kv_cache_size under the hood.
    """
    bytes_per_value = PRECISION_MAP.get(precision.lower(), 2)

    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = getattr(config, "hidden_size", None)

    if None in [num_layers, num_heads, hidden_size]:
        raise ValueError("Could not find required model configuration attributes.")

    head_dim = hidden_size // num_heads

    # Per token, per sequence, per batch:
    #   K: [num_kv_heads, head_dim]
    #   V: [num_kv_heads, head_dim]
    # Across all layers: * num_layers
    total_values = num_layers * num_kv_heads * head_dim * 2  # (K + V)
    total_bytes = total_values * bytes_per_value
    total_mb = total_bytes / (1024 * 1024)
    return total_mb

def detect_model_precision(model):
    """
    Returns one of: 'fp32', 'fp16', 'bf16', 'int8', 'int4'
    based on model dtype. (int8/int4 require HF quantization configs.)
    """

    # 1. HF models set this explicitly
    cfg_dtype = getattr(model.config, "torch_dtype", None)
    if cfg_dtype is not None:
        return dtype_to_precision(cfg_dtype)

    # 2. Check parameter dtype directly
    try:
        p = next(model.parameters())
        return dtype_to_precision(p.dtype)
    except StopIteration:
        pass

    # Fallback (very rare)
    return "fp16"


def dtype_to_precision(dtype):
    if dtype in (torch.float32, torch.float):
        return "fp32"
    if dtype in (torch.float16, torch.half):
        return "fp16"
    if dtype in (torch.bfloat16,):
        return "bf16"
    if dtype in (torch.uint8,):
        return "int8"   # quantized weight dtype in many HF quantized models

    # Unknown → default to fp16 as safe
    return "fp16"

def estimate_max_new_tokens_from_inputs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_beams: int = 1,
    free_mem_bytes: int = None,
    output_logits: bool = True,
    output_scores: bool = True,
    safety_factor: float = 0.75,
    # logits/scores are often effectively fp32 on GPU
    logits_dtype: str = "fp32",
) -> int:
    """
    Practical estimate of max_new_tokens using input_ids/attention_mask.

    - Uses prompt length only to respect context length
    - Memory side: KV cache + optional logits/scores
    - Applies a safety_factor to avoid OOM
    """

    if free_mem_bytes is None:
        free_mem_bytes, _ = torch.cuda.mem_get_info()

    free_mem_mb = free_mem_bytes / (1024 * 1024)
    usable_mem_mb = free_mem_mb * safety_factor

    batch_size = input_ids.shape[0]

    # Effective prompt length (max over batch)
    if attention_mask is not None:
        prompt_lengths = attention_mask.sum(dim=-1)
        max_prompt_len = int(prompt_lengths.max().item())
    else:
        max_prompt_len = int(input_ids.shape[1])

    precision = detect_model_precision(model)
    kv_mb_per_token_per_seq = calculate_kv_cache_size_per_token(model, precision)
    # print(kv_mb_per_token_per_seq)
    # ---- logits / scores per generated token, per sequence ----

    vocab_size = getattr(model.config, "vocab_size", None)
    logits_mb_per_token_per_seq = 0.0
    scores_mb_per_token_per_seq = 0.0

    if vocab_size is not None:
        logits_bytes_per_val = PRECISION_MAP.get(logits_dtype.lower(), 4)
        per_token_logits_mb = (vocab_size * logits_bytes_per_val) / (1024 * 1024)

        if output_logits:
            logits_mb_per_token_per_seq = per_token_logits_mb
        if output_scores:
            scores_mb_per_token_per_seq = per_token_logits_mb  # rough approx

    per_token_mb_per_seq = (
        kv_mb_per_token_per_seq
        + logits_mb_per_token_per_seq
        + scores_mb_per_token_per_seq
    )

    active_sequences = batch_size * num_beams

    if per_token_mb_per_seq <= 0 or active_sequences <= 0:
        return 0

    max_new_by_mem = usable_mem_mb / (per_token_mb_per_seq * active_sequences)

    # Context window bound
    max_new_by_ctx = None
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is not None:
        max_new_by_ctx = max_ctx - max_prompt_len

    if max_new_by_ctx is not None:
        max_new_tokens = int(max(0, min(max_new_by_mem, max_new_by_ctx)))
    else:
        max_new_tokens = int(max_new_by_mem)

    return max_new_tokens

from transformers import LogitsProcessor

class StopAfterReturnLine(LogitsProcessor):
    """
    Once we detect a line that starts with '    return', we allow generation
    to continue until the next newline is produced, then force EOS.
    Works per-sequence inside a batch.
    """
    def __init__(self, tokenizer, eos_token_id=None, indent="    "):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.indent = indent

        # Per-sequence state (grows to batch size as needed)
        self.seen_return_line = []
        self.waiting_for_newline = []

    def _ensure_batch(self, batch_size: int):
        while len(self.seen_return_line) < batch_size:
            self.seen_return_line.append(False)
            self.waiting_for_newline.append(False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch, seq_len]
        bsz = input_ids.size(0)
        self._ensure_batch(bsz)

        # If a sequence already saw a return-line and just produced '\n', force EOS now
        # Otherwise, keep scanning for '\n    return' (start of a line with 4-space indent).
        for i in range(bsz):
            if self.waiting_for_newline[i]:
                # Check if the last generated character is a newline by decoding a tiny tail.
                tail_ids = input_ids[i, -8:].tolist()
                tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
                if "\n" in tail_text:  # newline has appeared; force eos this step
                    scores[i, :] = -float("inf")
                    scores[i, self.eos_token_id] = 0.0
                continue

            if not self.seen_return_line[i]:
                # Decode a limited tail for speed
                tail_ids = input_ids[i, -128:].tolist()
                tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)

                # Detect a return at the start of a line (newline + 4 spaces + return)
                if f"\n{self.indent}return" in tail_text:
                    self.seen_return_line[i] = True
                    self.waiting_for_newline[i] = True  # now wait until newline then EOS

        return scores


def batched_generate_legacy_old(prompts, model, tokenizer, batch_size=1, retokenizationp=0.0, **generation_kwargs):
    generations = []
    generation_tokens = []
    segmentor = generateSegments(tokenizer.vocab.keys())
    pbar = tqdm(total=len(prompts), desc="Generating")
    batch_size = batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",)

        res, batch_token_lengths = fiddle_tokens(batch_inputs.input_ids, batch_inputs.attention_mask, tokenizer, segmentor,
                            no_prefix_suffix=True, is_mcq=False, reason=False, retokenizationp=retokenizationp)

        res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in res.items()}

        # logits_processor = LogitsProcessorList([StopAfterReturnLine(tokenizer, eos_token_id=tokenizer.eos_token_id)])

        batch_outputs = model.generate(
            **res,
            num_return_sequences=1,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            # logits_processor=logits_processor,
            **generation_kwargs,
        )

        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        # remove the prompt from the generation
        #batch_generations = [gen[len(prompt) :] for prompt, gen in zip(batch_prompts, batch_generations)]
        #print(batch_generations)
        generations.extend(batch_generations)
        generation_tokens.extend(batch_outputs.sequences.tolist())
        pbar.update(len(batch_prompts))
        del res, batch_outputs
        gc.collect()
        torch.cuda.empty_cache()
    return {'generations':generations, 'generation_tokens':generation_tokens}

import hashlib
def batched_generate_legacy(
    prompts,
    model,
    tokenizer,
    batch_size=1,
    retokenizationp=0.0,
    return_last_hidden_states=False,
    hidden_states_dir=None,
    store_hidden_states_in_memory=False,
    hidden_states_pre_norm=False,
    **generation_kwargs,
):
    total_prompts = len(prompts)
    generations = [None] * total_prompts
    generation_tokens = [None] * total_prompts
    generation_hidden_state_hashes = [None] * total_prompts if return_last_hidden_states else None
    generation_hidden_states = [None] * total_prompts if store_hidden_states_in_memory else None
    segmentor = generateSegments(tokenizer.vocab.keys())
    existing_hidden_hashes = set()
    hidden_hash_to_file = {}
    index_path = None
    sequence_index_path = None
    existing_sequences = {}
    existing_sequence_files = set()
    if return_last_hidden_states and hidden_states_dir is not None:
        os.makedirs(hidden_states_dir, exist_ok=True)
        index_path = os.path.join(hidden_states_dir, "hidden_state_index.jsonl")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as fi:
                    for line in fi:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        hash_val = record.get("hash")
                        file_val = record.get("file")
                        if hash_val:
                            existing_hidden_hashes.add(hash_val)
                            if file_val:
                                hidden_hash_to_file[hash_val] = file_val
            except OSError:
                pass
    if hidden_states_dir is not None:
        os.makedirs(hidden_states_dir, exist_ok=True)
        sequence_index_path = os.path.join(hidden_states_dir, "generated_sequences_index.jsonl")
        if os.path.exists(sequence_index_path):
            try:
                with open(sequence_index_path, "r") as fi:
                    for line in fi:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        prompt_idx = record.get("prompt_idx")
                        file_val = record.get("file")
                        if isinstance(prompt_idx, int) and file_val:
                            existing_sequence_files.add(file_val)
            except OSError:
                pass
        else:
            try:
                for fname in os.listdir(hidden_states_dir):
                    if fname.startswith("generated_sequences_batch_") and fname.endswith(".jsonl"):
                        existing_sequence_files.add(fname)
            except OSError:
                pass

    model_id = None
    if return_last_hidden_states:
        try:
            model_id = model.config._name_or_path
        except Exception:
            model_id = None
        if not model_id:
            model_id = getattr(model, "name_or_path", "") or "unknown_model"
    if existing_sequence_files:
        for fname in sorted(existing_sequence_files):
            batch_path = os.path.join(hidden_states_dir, fname)
            try:
                with open(batch_path, "r") as fi:
                    for line in fi:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        prompt_idx = record.get("prompt_idx")
                        if not isinstance(prompt_idx, int):
                            continue
                        if prompt_idx < 0 or prompt_idx >= total_prompts:
                            continue
                        existing_sequences[prompt_idx] = record
            except OSError:
                continue

    for prompt_idx, record in existing_sequences.items():
        gen = record.get("generation")
        toks = record.get("generation_tokens")
        if gen is None or toks is None:
            continue
        generations[prompt_idx] = gen
        generation_tokens[prompt_idx] = toks
        if return_last_hidden_states:
            hash_val = record.get("hash")
            if not hash_val:
                try:
                    hasher = hashlib.sha256()
                    hasher.update(model_id.encode("utf-8"))
                    hasher.update(b"|")
                    hasher.update(bytes(memoryview(np.asarray(toks, dtype=np.int32))))
                    hash_val = hasher.hexdigest()
                except Exception:
                    hash_val = None
            generation_hidden_state_hashes[prompt_idx] = hash_val
        if store_hidden_states_in_memory and return_last_hidden_states:
            hash_val = generation_hidden_state_hashes[prompt_idx]
            file_val = hidden_hash_to_file.get(hash_val) if hash_val else None
            if file_val:
                batch_path = os.path.join(hidden_states_dir, file_val)
                try:
                    data = np.load(batch_path)
                    hashes = data.get("hashes")
                    hidden = data.get("hidden_states")
                    if hashes is not None and hidden is not None:
                        match = np.where(hashes == hash_val)[0]
                        if match.size > 0:
                            generation_hidden_states[prompt_idx] = hidden[int(match[0])].tolist()
                except OSError:
                    pass

    pending_indices = [i for i in range(total_prompts) if generations[i] is None]
    pbar = tqdm(total=total_prompts, desc="Generating")
    if total_prompts > 0 and (total_prompts - len(pending_indices)) > 0:
        pbar.update(total_prompts - len(pending_indices), )
        print(f"Resuming generation: {len(pending_indices)} / {total_prompts} prompts remaining.")

    def _next_batch_id(prefix, suffix):
        max_id = -1
        try:
            for fname in os.listdir(hidden_states_dir):
                if not (fname.startswith(prefix) and fname.endswith(suffix)):
                    continue
                core = fname[len(prefix) : -len(suffix)]
                digits = ""
                for ch in core:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    max_id = max(max_id, int(digits))
        except OSError:
            pass
        return max_id + 1

    next_seq_batch_id = None
    next_hidden_batch_id = None
    if hidden_states_dir is not None:
        next_seq_batch_id = _next_batch_id("generated_sequences_batch_", ".jsonl")
        next_hidden_batch_id = _next_batch_id("hidden_states_batch_", ".npz")

    for start in range(0, len(pending_indices), batch_size):
        batch_prompt_indices = pending_indices[start : start + batch_size]
        batch_prompts = [prompts[idx] for idx in batch_prompt_indices]

        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",)

        res, batch_token_lengths = fiddle_tokens(batch_inputs.input_ids, batch_inputs.attention_mask, tokenizer, segmentor,
                            no_prefix_suffix=True, is_mcq=False, reason=False, retokenizationp=retokenizationp)

        res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in res.items()}

        # logits_processor = LogitsProcessorList([StopAfterReturnLine(tokenizer, eos_token_id=tokenizer.eos_token_id)])

        batch_outputs = model.generate(
            **res,
            num_return_sequences=1,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            # logits_processor=logits_processor,
            **generation_kwargs,
        )

        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
        # remove the prompt from the generation
        #batch_generations = [gen[len(prompt) :] for prompt, gen in zip(batch_prompts, batch_generations)]
        #print(batch_generations)
        batch_sequences = batch_outputs.sequences.tolist()
        for local_idx, prompt_idx in enumerate(batch_prompt_indices):
            generations[prompt_idx] = batch_generations[local_idx]
            generation_tokens[prompt_idx] = batch_sequences[local_idx]

        if return_last_hidden_states:
            batch_hashes = []
            for seq in batch_sequences:
                hasher = hashlib.sha256()
                hasher.update(model_id.encode("utf-8"))
                hasher.update(b"|")
                hasher.update(bytes(memoryview(np.asarray(seq, dtype=np.int32))))
                batch_hashes.append(hasher.hexdigest())
            for local_idx, prompt_idx in enumerate(batch_prompt_indices):
                generation_hidden_state_hashes[prompt_idx] = batch_hashes[local_idx]

            should_compute_hidden = True
            if batch_hashes:
                should_compute_hidden = not all(h in existing_hidden_hashes for h in batch_hashes)

            if should_compute_hidden:
                seqs = batch_outputs.sequences
                input_len = res["input_ids"].shape[1]
                gen_len = seqs.shape[1] - input_len
                if "attention_mask" in res:
                    attn = res["attention_mask"]
                else:
                    attn = torch.ones_like(res["input_ids"])
                if gen_len > 0:
                    gen_mask = torch.ones(
                        (seqs.shape[0], gen_len),
                        device=attn.device,
                        dtype=attn.dtype,
                    )
                    attn = torch.cat([attn, gen_mask], dim=1)
                seqs = seqs.to(model.device, non_blocking=True)
                attn = attn.to(model.device, non_blocking=True)
                captured_pre_norm = None
                hook_handle = None
                if hidden_states_pre_norm:
                    if hasattr(model, "model") and hasattr(model.model, "norm"):
                        def _capture_pre_norm(_, inputs, __):
                            nonlocal captured_pre_norm
                            if inputs:
                                captured_pre_norm = inputs[0].detach().cpu()

                        hook_handle = model.model.norm.register_forward_hook(_capture_pre_norm)
                    else:
                        raise ValueError("Model does not have a pre-norm layer to hook into.")

                with torch.inference_mode():
                    out = model(
                        seqs,
                        attention_mask=attn,
                        return_dict=True,
                        output_hidden_states=not hidden_states_pre_norm,
                    )
                    if hidden_states_pre_norm and captured_pre_norm is not None:
                        last_hidden = captured_pre_norm
                    else:
                        last_hidden = out.hidden_states[-1].detach().cpu()
                if hook_handle is not None:
                    hook_handle.remove()

                if hidden_states_dir is not None:
                    batch_path = os.path.join(
                        hidden_states_dir, f"hidden_states_batch_{next_hidden_batch_id:06d}.npz"
                    )
                    next_hidden_batch_id += 1
                    np.savez_compressed(
                        batch_path,
                        hashes=np.array(batch_hashes, dtype="U64"),
                        hidden_states=last_hidden.to(torch.float16).numpy(),
                    )
                    if index_path is not None:
                        with open(index_path, "a") as fo:
                            for h in batch_hashes:
                                if h in existing_hidden_hashes:
                                    continue
                                fo.write(json.dumps({"hash": h, "file": os.path.basename(batch_path)}) + "\n")
                                existing_hidden_hashes.add(h)

                if store_hidden_states_in_memory:
                    batch_hidden = last_hidden.tolist()
                    for local_idx, prompt_idx in enumerate(batch_prompt_indices):
                        generation_hidden_states[prompt_idx] = batch_hidden[local_idx]
                del out, last_hidden

        if hidden_states_dir is not None:
            batch_seq_path = os.path.join(
                hidden_states_dir, f"generated_sequences_batch_{next_seq_batch_id:06d}.jsonl"
            )
            next_seq_batch_id += 1
            try:
                with open(batch_seq_path, "w") as fo:
                    for local_idx, prompt_idx in enumerate(batch_prompt_indices):
                        record = {
                            "prompt_idx": prompt_idx,
                            "generation": batch_generations[local_idx],
                            "generation_tokens": batch_sequences[local_idx],
                        }
                        if return_last_hidden_states:
                            record["hash"] = generation_hidden_state_hashes[prompt_idx]
                        fo.write(json.dumps(record) + "\n")
                if sequence_index_path is not None:
                    with open(sequence_index_path, "a") as fo:
                        for local_idx, prompt_idx in enumerate(batch_prompt_indices):
                            rec = {
                                "prompt_idx": prompt_idx,
                                "file": os.path.basename(batch_seq_path),
                            }
                            if return_last_hidden_states:
                                rec["hash"] = generation_hidden_state_hashes[prompt_idx]
                            fo.write(json.dumps(rec) + "\n")
            except OSError:
                pass

        pbar.update(len(batch_prompts))
        del res, batch_outputs
        gc.collect()
        torch.cuda.empty_cache()
    output = {'generations': generations, 'generation_tokens': generation_tokens}
    if return_last_hidden_states:
        output["generation_hidden_state_hashes"] = generation_hidden_state_hashes
        if store_hidden_states_in_memory:
            output["generation_hidden_states"] = generation_hidden_states
    return output

def hidden_states_from_sequences(
    sequences,
    model,
    tokenizer=None,
    batch_size=8,
    hidden_states_pre_norm=False,
):
    """
    Compute last hidden states for pre-tokenized sequences.
    `sequences` is a list of lists of token ids.
    """
    if sequences is None:
        return []
    if len(sequences) == 0:
        return []

    model.eval()
    device = model.device

    pad_token_id = None
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    outputs = []
    for start in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[start : start + batch_size]
        max_len = max(len(seq) for seq in batch)
        input_ids = torch.full(
            (len(batch), max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(batch), max_len),
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(batch):
            if not seq:
                continue
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        captured_pre_norm = None
        hook_handle = None
        if hidden_states_pre_norm:
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                def _capture_pre_norm(_, inputs, __):
                    nonlocal captured_pre_norm
                    if inputs:
                        captured_pre_norm = inputs[0].detach().cpu()

                hook_handle = model.model.norm.register_forward_hook(_capture_pre_norm)
            else:
                raise ValueError("Model does not have a pre-norm layer to hook into.")

        with torch.inference_mode():
            out = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=not hidden_states_pre_norm,
            )
            if hidden_states_pre_norm and captured_pre_norm is not None:
                last_hidden = captured_pre_norm
            else:
                last_hidden = out.hidden_states[-1].detach().cpu()
        if hook_handle is not None:
            hook_handle.remove()

        outputs.extend(last_hidden.tolist())
        del out, last_hidden
        gc.collect()
        torch.cuda.empty_cache()

    return outputs


def load_hidden_state_by_hash(hidden_states_dir, hash_val):
    """
    Load a single hidden state by hash from a hidden states directory.
    Expects hidden_state_index.jsonl plus hidden_states_batch_*.npz (or legacy .pt).
    Returns a numpy array or None if not found.
    """
    if not hidden_states_dir or not hash_val:
        return None
    index_path = os.path.join(hidden_states_dir, "hidden_state_index.jsonl")
    candidate_files = []
    if os.path.exists(index_path):
        # print('Found index path:', index_path)
        try:
            with open(index_path, "r") as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("hash") == hash_val and record.get("file"):
                        candidate_files.append(record["file"])
        except OSError:
            pass
    # if not candidate_files:
    #     try:
    #         for fname in os.listdir(hidden_states_dir):
    #             if fname.startswith("hidden_states_batch_") and (
    #                 fname.endswith(".npz") or fname.endswith(".pt")
    #             ):
    #                 candidate_files.append(fname)
    #     except OSError:
    #         return None
    if not candidate_files:
        print('No candidate files found in index')
        return None
    for fname in candidate_files:
        batch_path = os.path.join(hidden_states_dir, fname)
        if fname.endswith(".npz"):
            try:
                data = np.load(batch_path)
                hashes = data.get("hashes")
                hidden = data.get("hidden_states")
                if hashes is None or hidden is None:
                    continue
                match = np.where(hashes == hash_val)[0]
                if match.size > 0:
                    return hidden[int(match[0])]
            except OSError:
                continue
        elif fname.endswith(".pt"):
            try:
                data = torch.load(batch_path, map_location="cpu")
                hashes = data.get("hashes")
                hidden = data.get("hidden_states")
                if hashes is None or hidden is None:
                    continue
                hashes = np.asarray(hashes, dtype="U64")
                match = np.where(hashes == hash_val)[0]
                if match.size > 0:
                    if isinstance(hidden, torch.Tensor):
                        return hidden[int(match[0])].detach().cpu().numpy()
                    return np.asarray(hidden)[int(match[0])]
            except OSError:
                continue
        return None

def load_humaneval(model_name, base_dir=None, sampling_size=51, typos=False):
    if base_dir is None:
        if typos:
            base_dir = '/scratch/kjain25/TokenizationProject/results_passattypos/humaneval/'
        else:
            base_dir = '/scratch/kjain25/Tokenizer_passK/results_passretok/humaneval/'
    base_dir = os.path.join(base_dir) + f'/{model_name.replace("/", "_")}/'

    if os.path.exists(base_dir+f'retokp_maxexamples_164_unbiasedsize_{sampling_size}_df_data.df'):
        print('Loading pre-existing processed data files...')
        df_data = pd.read_hdf(base_dir+f'retokp_maxexamples_164_unbiasedsize_{sampling_size}_df_data.df', key='df')
        df_data_0 = pd.read_hdf(base_dir+f'retokp_maxexamples_164_unbiasedsize_{sampling_size}_df_data_0.df', key='df')
    else:
        if typos:
            pstr = f'typop_%0.1f_maxexamples_164_unbiasedsize_{sampling_size}'
        else:
            pstr = f'retokp_%0.1f_maxexamples_164_unbiasedsize_{sampling_size}'
        pvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        print('Reading raw data files and processing prompt_tokenization_length...')

        if not os.path.exists(f'{base_dir}/'+pstr%0.0+'/scored_predictions.jsonl'):
            raise FileNotFoundError(f"Expected file not found: {base_dir}/"+pstr%0.0+'/scored_predictions.jsonl')
        df_data_0 = pd.read_json(f'{base_dir}/'+pstr%0.0+'/scored_predictions.jsonl', lines=True)
        print(df_data_0.shape)
        dfs2 = []
        for p in pvals[:]:
            print(f'Reading p={p:0.2f}', end='\r')
            if p==0.0:
                df2 = pd.read_json(f'{base_dir}/'+pstr%p+'/dontsample/scored_predictions.jsonl', lines=True)
            else:
                df2 = pd.read_json(f'{base_dir}/'+pstr%p+'/scored_predictions.jsonl', lines=True)
            df2['p'] = p
            dfs2.append(df2)
            print(df2.shape)
        df_data = pd.concat(dfs2, ignore_index=True)

        df_data['prompt_tokenization_length'] = 0
        df_data['prompt_tokenization_index'] = 0


        tokenizer = load_tokenizer(model_name)

        for i in tqdm(range(len(df_data)), desc='Computing prompt tokenization lengths'):
            prompt = (df_data.prompt.values[i])
            for j in range(len(tokenizer.encode(prompt)),len(df_data.generation_tokens.values[i])+1):
                if tokenizer.decode(df_data.generation_tokens.values[i][:j], skip_special_tokens=True) == prompt:
                    df_data.loc[i, 'prompt_tokenization_length'] = len([tok_ for tok_ in df_data.generation_tokens.values[i][:j] if tok_ not in tokenizer.all_special_ids])
                    df_data.loc[i, 'prompt_tokenization_index'] = j
                    break

            if j == len(df_data.generation_tokens.values[i]):
                print('Not found for index', i)
                break

        df_data_0['prompt_tokenization_length'] = 0
        df_data_0['prompt_tokenization_index'] = 0

        for t in df_data_0.task_id.unique():
            prompt = df_data_0[df_data_0.task_id==t].prompt.values[0]
            df_data_0.loc[df_data_0.task_id==t, 'prompt_tokenization_length'] = len(tokenizer.encode(prompt))

        for i in tqdm(range(len(df_data_0)), desc='Computing prompt tokenization lengths for p=0.0'):
            prompt = (df_data_0.prompt.values[i])
            for j in range(len(tokenizer.encode(prompt)),len(df_data_0.generation_tokens.values[i])+1):
                if tokenizer.decode(df_data_0.generation_tokens.values[i][:j], skip_special_tokens=True) == prompt:
                    df_data_0.loc[i, 'prompt_tokenization_index'] = j
                    break

        df_data['Correct'] = df_data['passed'].astype(int)
        df_data_0['Correct'] = df_data_0['passed'].astype(int)
        df_data.to_hdf(f'{base_dir}/retokp_maxexamples_164_unbiasedsize_{sampling_size}_df_data.df', key='df', mode='w')
        df_data_0.to_hdf(f'{base_dir}/retokp_maxexamples_164_unbiasedsize_{sampling_size}_df_data_0.df', key='df', mode='w')

    return df_data, df_data_0

def load_gsm8k_python(model_name, base_dir=None, sampling_size=11, dataset_size_N = 1000, typos=False):
    if base_dir is None:
        if typos:
            base_dir = '/scratch/kjain25/TokenizationProject/results_passattypos/gsm8k_python/'
        else:
            base_dir = '/scratch/kjain25/Tokenizer_passK/results_passretok/gsm8k_python/'
    base_dir = os.path.join(base_dir) + f'/{model_name.replace("/", "_")}/'

    if os.path.exists(base_dir+f'retokp_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}_df_data.df'):
        print('Loading pre-existing processed data files...')
        df_data = pd.read_hdf(base_dir+f'retokp_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}_df_data.df', key='df')
        df_data_0 = pd.read_hdf(base_dir+f'retokp_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}_df_data_0.df', key='df')
    else:
        if typos:
            pstr = f'typop_%0.1f_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}'
        else:
            pstr = f'retokp_%0.1f_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}'
        pvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        print('Reading raw data files and processing prompt_tokenization_length...')

        if not os.path.exists(f'{base_dir}/'+pstr%0.0+'/scored_predictions.jsonl'):
            raise FileNotFoundError(f"Expected file not found: {base_dir}/"+pstr%0.0+'/scored_predictions.jsonl')
        df_data_0 = pd.read_json(f'{base_dir}/'+pstr%0.0+'/scored_predictions.jsonl', lines=True)
        print(df_data_0.shape)
        dfs2 = []
        for p in pvals[:]:
            print(f'Reading p={p:0.2f}', end='\r')
            if p==0.0:
                df2 = pd.read_json(f'{base_dir}/'+pstr%p+'/dontsample/scored_predictions.jsonl', lines=True)
            else:
                df2 = pd.read_json(f'{base_dir}/'+pstr%p+'/scored_predictions.jsonl', lines=True)
            df2['p'] = p
            dfs2.append(df2)
            print(df2.shape)
        df_data = pd.concat(dfs2, ignore_index=True)

        df_data['prompt_tokenization_length'] = 0
        df_data['prompt_tokenization_index'] = 0


        tokenizer = load_tokenizer(model_name)

        for i in tqdm(range(len(df_data)), desc='Computing prompt tokenization lengths'):
            prompt = (df_data.prompt.values[i])
            for j in range(len(tokenizer.encode(prompt)),len(df_data.generation_tokens.values[i])+1):
                if tokenizer.decode(df_data.generation_tokens.values[i][:j], skip_special_tokens=True) == prompt:
                    df_data.loc[i, 'prompt_tokenization_length'] = len([tok_ for tok_ in df_data.generation_tokens.values[i][:j] if tok_ not in tokenizer.all_special_ids])
                    df_data.loc[i, 'prompt_tokenization_index'] = j
                    break

            if j == len(df_data.generation_tokens.values[i]):
                print('Not found for index', i)
                break

        df_data_0['prompt_tokenization_length'] = 0
        df_data_0['prompt_tokenization_index'] = 0

        for t in df_data_0.task_id.unique():
            prompt = df_data_0[df_data_0.task_id==t].prompt.values[0]
            df_data_0.loc[df_data_0.task_id==t, 'prompt_tokenization_length'] = len(tokenizer.encode(prompt))

        for i in tqdm(range(len(df_data_0)), desc='Computing prompt tokenization lengths for p=0.0'):
            prompt = (df_data_0.prompt.values[i])
            for j in range(len(tokenizer.encode(prompt)),len(df_data_0.generation_tokens.values[i])+1):
                if tokenizer.decode(df_data_0.generation_tokens.values[i][:j], skip_special_tokens=True) == prompt:
                    df_data_0.loc[i, 'prompt_tokenization_index'] = j
                    break

        df_data['Correct'] = df_data['passed'].astype(int)
        df_data_0['Correct'] = df_data_0['passed'].astype(int)
        df_data.to_hdf(f'{base_dir}/retokp_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}_df_data.df', key='df', mode='w')
        df_data_0.to_hdf(f'{base_dir}/retokp_maxexamples_{dataset_size_N}_unbiasedsize_{sampling_size}_df_data_0.df', key='df', mode='w')

    return df_data, df_data_0

def load_gsm8k(model_name, base_dir=None, typos=False, sampling_size=30):
    if base_dir is None:
        if typos:
            base_dir = '/scratch/kjain25/TokenizationProject/results_passattypos/gsm8k/'
        else:
            base_dir = '/scratch/kjain25/Tokenizer_passK/results_passretok/gsm8k/'
    base_dir = os.path.join(base_dir) + f'/{model_name.replace("/", "_")}/'

    filestr = f'gsm8k_N_1000_numretokenizations_{sampling_size}' if not typos else f'gsm8k_N_1000_numvariants_{sampling_size}_typop'

    if os.path.exists(base_dir+f'{filestr}_df_data.df'):
        print('Loading pre-existing processed data files...')
        df_data = pd.read_hdf(base_dir+f'{filestr}_df_data.df', key='df')
        df_data_0 = pd.read_hdf(base_dir+f'{filestr}_df_data_0.df', key='df')
    else:
        pstr = f'{filestr}_%0.2f'
        pvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        print('Reading raw data files and processing prompt_tokenization_length...')

        df_data_0 = pd.read_hdf(f'{base_dir}/'+pstr%0.0+'_sampled.df', )
        print(df_data_0.shape)
        dfs2 = []
        for p in pvals[:]:
            print(f'Reading p={p:0.2f}',)
            df2 = pd.read_hdf(f'{base_dir}/'+pstr%p+'.df',)
            dfs2.append(df2)
        df_data = pd.concat(dfs2, ignore_index=True)

        df_data.to_hdf(f'{base_dir}/{filestr}_df_data.df', key='df', mode='w')
        df_data_0.to_hdf(f'{base_dir}/{filestr}_df_data_0.df', key='df', mode='w')

    return df_data, df_data_0

def load_mmlu(model_name, base_dir=None, typos=False, sampling_size=30):
    if typos:
        filestr = f'mmlu_N_1000_numvariants_{sampling_size}_typop'
    else:
        filestr = f'mmlu_N_1000_numretokenizations_{sampling_size}'
    if base_dir is None:
        if typos:
            base_dir = '/scratch/kjain25/TokenizationProject/results_passattypos/mmlu/'
            base_dir = base_dir + f'/{model_name.replace("/", "_")}/'
        else:
            base_dir = "/scratch/kjain25/Tokenizer_passK/results_passretok/mmlu/"

            base_dir = base_dir + f'/{model_name.replace("/", "_")}/without_reasoning'

    if os.path.exists(base_dir+f'/df_data_temp1.00.h5'):
        df_data = pd.read_hdf(base_dir + '/df_data_temp1.00.h5',)
        return df_data

    else:
        print(base_dir+f'/df_data_temp1.00.df' + ' not found, loading from raw files and processing...')
        dfs = []
        for pvals in [0.00,0.20,0.4,0.6,0.8,1.0]:
            dfs.append(pd.read_hdf(base_dir + f'/{filestr}_{pvals:.2f}.df',))
        df_data = pd.concat(dfs, ignore_index=True)
        return df_data

    return df_data



## TODO
"""
 1. Implement temperature sampling in batched_generate. - add option to do different sampling methods.
# 2. Correct generateSegments to make sure cache is properly used
# 3. Add storing of x entropy, picked token values of logit, unscaled and temperature-scaled probabilities, and partition function.
# 4. Add option to fiddle only certain parts of the prompt instead of the whole prompt - add p value metric.
# 5. Add pass at k!! That's the most important part.
# 6. add ability to parallelize to run on multiple GPUs - parallelize tasks, maybe?
# 7. Add ability to allow reasoning, changing the prompts to make sure the model can reason before answering.
8. save after each prompt or batch is processed, in case of crashes.
9. save selected prompts.

"""

"""
So each run is characterized by:

1. Model name 
2. Dataset 
3. retokenization p-values (or range of values??)
~~4. retokenizations per p value~~
5. Temperature of sampling 
6. Method of sampling 
7. Reasoning method - long, short, mid? 
8. S for pass@k
9. 

"""

# model_name/dataset_name/promptN_pval_samples_topk_temperature.jsonl