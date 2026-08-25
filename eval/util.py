import gc
import hashlib
import json
import os
import random
from string import ascii_uppercase

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from methodtools import lru_cache
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging

from eval.typo import TYPO_REPLACEMENTS, fiddle_tokens_typos, randomly_replace_letter

logging.set_verbosity_error()




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
        chosen = random.choices(choices, weights=weights, k=1)[0]
        return [chosen] + self.buildSegments(word, start + len(chosen))

    def generate(self, word):
        return self.buildSegments(word, 0)


def prep_incontext_examples(test_df, num_incontext_examples):
    indices = np.arange(len(test_df))
    incontext_indices = {
        i: np.random.choice(indices[indices != i], size=num_incontext_examples, replace=False)
        for i in tqdm(indices, desc="Precomputing in-context examples")
    }
    return incontext_indices


def format_example(
    question, passage=None, choices=None, answer=None, qa_format="qnan", question_prefix="Question:", answer_prefix="Answer:"
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


def get_checkpoints(model_name):
    refs = HfApi().list_repo_refs(model_name)
    checkpoints = []
    for branch in refs.branches:
        checkpoints.append(branch.name)
    return checkpoints


def _get_prompt_wrapper_ids(tokenizer, is_mcq, reason, no_prefix_suffix=False):
    if no_prefix_suffix:
        return [], []

    if is_mcq:
        if reason:
            input_ids_prefix = tokenizer.encode(
                """<|im_start|>system\n Answer the following multiple choice question. The last line of your response should be of the following format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering. <|im_end|>\n<|im_start|>user\n""",
                add_special_tokens=False,
            )
        else:
            input_ids_prefix = tokenizer.encode(
                """<|im_start|>system\nYou are a helpful assistant. For the following multiple choice questions, return the answer only, without any additional reasoning or explanation. <|im_end|>\n<|im_start|>user\n""",
                add_special_tokens=False,
            )
    else:
        input_ids_prefix = tokenizer.encode(
            """<|im_start|>system\nYou are a helpful assistant. <|im_end|>\n<|im_start|>user\n""",
            add_special_tokens=False,
        )

    input_ids_suffix = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    return input_ids_prefix, input_ids_suffix


def _extract_real_prompt_token_ids(prompt_ids, attention_row, tokenizer):
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    real_token_ids = []
    for token_id, attended in zip(prompt_ids, attention_row):
        if int(attended) != 1:
            continue
        if bos_token_id is not None and token_id == bos_token_id:
            continue
        real_token_ids.append(int(token_id))
    return real_token_ids


def fiddle_tokens(inputTensor, attention_mask, tokenizer, segmentor, is_mcq, reason, retokenizationp, no_prefix_suffix=False):
    inputList = inputTensor.tolist()
    fiddledPrompts = []

    input_ids_prefix, input_ids_suffix = _get_prompt_wrapper_ids(
        tokenizer,
        is_mcq=is_mcq,
        reason=reason,
        no_prefix_suffix=no_prefix_suffix,
    )

    lengths = []
    for i, prompt in enumerate(inputList):
        fiddledPrompt = []
        realTokenIds = _extract_real_prompt_token_ids(prompt, attention_mask[i], tokenizer)
        realTokens = tokenizer.convert_ids_to_tokens(realTokenIds)
        for token in realTokens:
            if retokenizationp == 0.0 or (random.random() >= retokenizationp):
                fiddledPrompt.append(token)
            else:
                fiddledPrompt += segmentor.generate(token)
        lengths.append((len(input_ids_prefix), len(fiddledPrompt), len(input_ids_suffix)))
        fiddledPrompts.append(input_ids_prefix + tokenizer.convert_tokens_to_ids(fiddledPrompt) + input_ids_suffix)

    return tokenizer.pad({"input_ids": fiddledPrompts}, padding="longest", padding_side="left", return_tensors="pt"), lengths


def load_tokenizer(tokenizer_name_or_path, padding_side="left"):
    print(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.backend_tokenizer.model.dropout = 0.0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model(model_name_or_path, step=None, checkpoint=None):
    revision = None
    if step is not None and checkpoint is not None:
        raise ValueError("Specify only one of `step` or `checkpoint`.")

    if os.path.exists(model_name_or_path):
        if checkpoint is not None:
            model_name_or_path += f"/{checkpoint}"
        elif step is not None:
            model_name_or_path += f"/step{step}"
    else:
        if checkpoint is not None:
            available_checkpoints = get_checkpoints(model_name_or_path)
            if checkpoint not in available_checkpoints:
                raise ValueError(f"Checkpoint {checkpoint} not found")
            revision = checkpoint
            print(f"Revision: {revision}")
        elif step is not None:
            try:
                ckps = [r for r in get_checkpoints(model_name_or_path) if len(r.split("-")) > 1]
                revision = [r for r in ckps if r.split("-")[1] == f"step{step}"][0]
                print(f"Revision: {revision}")
            except IndexError:
                raise ValueError(f"Checkpoint {step} not found")

    print(f"Loading model from {model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        revision=revision if "allenai" in model_name_or_path else None,
    )
    model.eval()

    return model


def load_model_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    step=None,
    checkpoint=None,
    padding_side="left",
):
    model = load_model(model_name_or_path, step=step, checkpoint=checkpoint)
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = load_tokenizer(tokenizer_name_or_path, padding_side=padding_side)
    return model, tokenizer


def batched_generate_legacy(
    prompts,
    model,
    tokenizer,
    batch_size=1,
    retokenizationp=0.0,
    perturbation_mode="retokenization",
    typo_p=0.0,
    typo_region="all",
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
    segmentor = generateSegments(tokenizer.vocab.keys()) if perturbation_mode == "retokenization" else None
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
        pbar.update(total_prompts - len(pending_indices))
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
            padding="longest",
        )

        if perturbation_mode == "retokenization":
            res, batch_token_lengths = fiddle_tokens(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                tokenizer,
                segmentor,
                no_prefix_suffix=True,
                is_mcq=False,
                reason=False,
                retokenizationp=retokenizationp,
            )
        elif perturbation_mode == "typo":
            res, batch_token_lengths = fiddle_tokens_typos(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                tokenizer,
                no_prefix_suffix=True,
                is_mcq=False,
                reason=False,
                typo_p=typo_p,
                typo_region=typo_region,
            )
        else:
            raise ValueError(f"Unknown perturbation_mode: {perturbation_mode}")

        res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in res.items()}

        batch_outputs = model.generate(
            **res,
            num_return_sequences=1,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )

        batch_generations = tokenizer.batch_decode(batch_outputs.sequences, skip_special_tokens=True)
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
        del res, batch_outputs, batch_token_lengths
        gc.collect()
        torch.cuda.empty_cache()

    output = {"generations": generations, "generation_tokens": generation_tokens}
    if return_last_hidden_states:
        output["generation_hidden_state_hashes"] = generation_hidden_state_hashes
        if store_hidden_states_in_memory:
            output["generation_hidden_states"] = generation_hidden_states
    return output
