import os, sys, argparse
from pathlib import Path

# Single CLI parser at the very top (parse before importing torch)
parser = argparse.ArgumentParser(description="Run passretok with configurable GPU(s) and p-value(s).")
parser.add_argument("--gpus", type=str, help="GPU id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--p", type=float, help="Single retokenization probability, e.g. 0.4.")
parser.add_argument("--pvals", type=str, help="Comma-separated retokenization probabilities, e.g. '0.0,0.2,0.4'.")
parser.add_argument("--temp_ident", type=str, help="Temporary identifier to add", default="")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Model name or path.",
    default="allenai/OLMo-2-1124-7B-Instruct",
)
args = parser.parse_args()

# Set CUDA visibility from --gpus (default remains '1')
if args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import pickle

import gc
import re
import torch
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from scipy.special import comb
import tqdm
from eval import util
from eval.paths import get_results_dataset_dir
import contextlib


def parse_generated_answer(decoded_text):
    boxed_matches = re.findall(r"\\boxed\{([ABCD])\}", decoded_text)
    if boxed_matches:
        return boxed_matches[-1]

    letter_matches = re.findall(r"\b([ABCD])\b", decoded_text.upper())
    if letter_matches:
        return letter_matches[-1]

    return None


def decode_generated_answer(generated_token_ids, tokenizer, reasoning):
    if len(generated_token_ids) == 0:
        return None
    if not reasoning:
        return tokenizer.decode([generated_token_ids[0]], skip_special_tokens=True).strip() or None
    decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return parse_generated_answer(decoded_text)

output_dir = str(get_results_dataset_dir("retok", "mmlu"))

model_name = args.model_name_or_path
reasoning = False
dataset = 'mmlu'
N = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "eval" / dataset / f"sampled_data_{N}_sorted.h5"
RAW_DATA_PATH = PROJECT_ROOT / "eval" / dataset / "test.jsonl"

no_prefix_suffix = False # this is a argument in fiddle_tokens
if reasoning:
    answer_prefix = 'Answer and Reasoning:'
else:
    answer_prefix = 'Answer:'
max_new_tokens = 512 if reasoning else 1

# Build p-values from CLI (use args parsed above)
if args.pvals:
    p_vals = [float(x) for x in args.pvals.split(",") if x.strip() != ""]
elif args.p is not None:
    p_vals = [float(args.p)]
else:
    p_vals = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]

n_retokenizations = 11
n_retokenizations_0 = 1


if DATA_PATH.exists():
    test_df = pd.read_hdf(DATA_PATH, key='df')
else:
    test_df = pd.read_json(RAW_DATA_PATH, lines=True)
    test_df = test_df.sample(min(len(test_df), N), random_state=42).reset_index(drop=True)
    test_df["temp_len"] = test_df["question"].apply(len)
    test_df.sort_values("temp_len", inplace=True, ascending=False)
    test_df.drop(columns="temp_len", inplace=True)
    test_df.to_hdf(DATA_PATH, key='df', mode='w')
test_df = test_df.reset_index(drop=True)

model, tokenizer = util.load_model_and_tokenizer(model_name)

num_incontext_examples = 0


incontext_indices = util.prep_incontext_examples(test_df, num_incontext_examples)

prompts = []
prompts0 = []

prompts_i = []
prompts_i_0 = []

answers = []
answers_0 = []
task_ids = []
task_ids_0 = []

for i, row in test_df.iterrows():
    prompt = ""
    for j in incontext_indices[i]:
        ic_row = test_df.iloc[j]
        prompt += (
            util.format_example(
                ic_row["question"].strip(),
                choices=ic_row["choices"],
                answer="ABCD"[ic_row["answer"]],
                qa_format='False',
            )
            + "\n\n"
        )
    prompt += util.format_example(row["question"].strip(), choices=row["choices"], qa_format='False',
                                  answer_prefix=answer_prefix)
    for j in range(n_retokenizations):
        prompts.append(prompt)
        prompts_i.append(i)
        answers.append("ABCD"[row["answer"]])
        task_ids.append(f"mmlu/{i}")
    for j in range(n_retokenizations_0):
        prompts0.append(prompt)
        prompts_i_0.append(i)
        answers_0.append("ABCD"[row["answer"]])
        task_ids_0.append(f"mmlu/{i}")

segmentor = util.generateSegments(tokenizer.get_vocab().keys())

batch_size = {3:16, 50:32, 100:64, 200:128}
chkpoint = 1000

# Build a base prefix and create the output directory once
if reasoning:
    output_dir = f'{output_dir}/reasoning'

output_dir = f'{output_dir}/{model_name.replace("/","_")}/'
base_prefix = f'{output_dir}/{dataset}_N_{N}_numretokenizations_{n_retokenizations}'
os.makedirs(output_dir, exist_ok=True)

# Per-p state is initialized inside the p-loop
df_data = None
filename = None
data = []

# Helper to flush in-memory data to disk and clear data buffer.
def flush_data_chunk():
    global df_data, data, filename
    if len(data) == 0:
        return
    chunk_df_data = pd.DataFrame.from_records(data)
    if df_data is None or df_data.empty:
        df_data = chunk_df_data
    else:
        df_data = pd.concat([df_data, chunk_df_data], ignore_index=True)
    df_data.to_hdf(filename, mode='w', key='df')
    data.clear()

def select_batch_size(prompti: int) -> int:
    # If prompti is < key, use corresponding value; else use the largest value in the schedule.
    for thresh in sorted(batch_size.keys()):
        if prompti < thresh:
            return batch_size[thresh]
    return batch_size[sorted(batch_size.keys())[-1]]

gc.collect()
torch.cuda.empty_cache()

for retokp in p_vals[::-1]:
    # Set per-p file and load existing progress (if any)
    filename = f'{base_prefix}_{retokp:.2f}{args.temp_ident}.df'


    df_data = pd.read_hdf(filename, key='df') if os.path.exists(filename) else None
    data.clear()

    # Compute resume start index for this p based on existing rows.
    if df_data is not None and not df_data.empty:
        start_prompti = df_data.shape[0]
        print('Resuming from prompt index', start_prompti, 'for p=', retokp)
    else:
        start_prompti = 0

    new_added = 0
    total_prompts = len(prompts0) if retokp == 0.0 else len(prompts)

    prompti = start_prompti
    with tqdm.tqdm(total=total_prompts - start_prompti, desc=f'Processing p={retokp:0.2f}') as pbar:
        while prompti < total_prompts:
            # Periodically flush accumulated data to disk.
            if new_added >= chkpoint:
                flush_data_chunk()
                new_added = 0


            prompti_ = prompts_i_0[prompti] if retokp == 0.0 else prompts_i[prompti]
            bs = select_batch_size(prompti_)
            #update pbar description to also include bs
            pbar.set_description(f'Processing p={retokp:0.2f}, bs={bs}')


            if retokp == 0.0:
                generation_kwargs = {'num_return_sequences':1,
                                     'max_new_tokens':max_new_tokens,
                                     'do_sample':False,
                                     'return_dict_in_generate':True,
                                     'pad_token_id':tokenizer.pad_token_id,
                                     'tokenizer':tokenizer,
                                     }
                batch_prompts = prompts0[prompti: prompti + bs]
                batch_prompts_i = prompts_i_0[prompti: prompti + bs]
                batch_answers = answers_0[prompti: prompti + bs]
                batch_task_ids = task_ids_0[prompti: prompti + bs]
            else:
                generation_kwargs = {'num_return_sequences':1,
                                     'max_new_tokens':max_new_tokens,
                                     'do_sample':False,
                                     'return_dict_in_generate':True,
                                     'pad_token_id':tokenizer.pad_token_id,
                                     'tokenizer':tokenizer}
                batch_prompts = prompts[prompti: prompti + bs]
                batch_prompts_i = prompts_i[prompti: prompti + bs]
                batch_answers = answers[prompti: prompti + bs]
                batch_task_ids = task_ids[prompti: prompti + bs]

            if len(batch_prompts) == 0:
                break

            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                add_special_tokens=True,
                padding="longest",
            )

            # Still CPU here
            res, batch_token_lengths = util.fiddle_tokens(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                tokenizer,
                segmentor,
                no_prefix_suffix=no_prefix_suffix,
                retokenizationp=retokp,
                is_mcq=True,
                reason=reasoning,
            )
            if prompti == start_prompti:
                print('Sample fiddled input_ids:', tokenizer.decode(res['input_ids'][0], skip_special_tokens=True))
            use_autocast = (model.device.type == "cuda")
            res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v)
                           for k, v in res.items()}
            with torch.inference_mode():
                with (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                      if use_autocast else contextlib.nullcontext()):
                    logits_prompt = model(**res, return_dict=True).logits
                    probs_prompt = torch.softmax(logits_prompt[:, -1, :], dim=-1)
                    answer_idx = torch.tensor(
                        [tokenizer.convert_tokens_to_ids(answer) for answer in batch_answers],
                        device=probs_prompt.device,
                    ).unsqueeze(-1)
                    answer_probs = probs_prompt.gather(-1, answer_idx).squeeze(-1).cpu().tolist()
                    batch_outputs = model.generate(**res, **generation_kwargs)
                    len_inputs = res['input_ids'].shape[1]
                    for bi in range(len(batch_prompts)):
                        generated_token_ids = batch_outputs.sequences[bi][len_inputs:].tolist()
                        decoded_answer = decode_generated_answer(
                            generated_token_ids,
                            tokenizer=tokenizer,
                            reasoning=reasoning,
                        )
                        data.append({'task_id': batch_task_ids[bi],
                                     'p': retokp,
                                     'prompti': batch_prompts_i[bi],
                                     'prompt_tokens': batch_outputs.sequences[bi].tolist()[:len_inputs],
                                     'generated_tokens': generated_token_ids,
                                     'token_lengths': batch_token_lengths[bi],
                                     'answer': batch_answers[bi],
                                     'answer_prob': answer_probs[bi],
                                     'generated_answer': decoded_answer,
                                     'passed': int(decoded_answer == batch_answers[bi]),
                                     })

            del res, batch_outputs
            gc.collect()
            torch.cuda.empty_cache()
            step = len(batch_prompts)
            new_added += step
            prompti += step
            pbar.update(step)

    # Flush any remaining data for this p before moving to next.
    flush_data_chunk()

# Final check: df is up-to-date; optional summary remains unchanged.

# Build consolidated df across all p-values (from small to large)
# consolidated_filename = f"{base_prefix}.df"
# per_p_files = []
# for fname in os.listdir(output_dir):
#     # Expect filenames like: {base_prefix}_{p:.2f}.df
#     if fname.startswith(os.path.basename(base_prefix)) and fname.endswith(".df"):
#         try:
#             p_str = fname.replace(os.path.basename(base_prefix) + "_", "").replace(".df", "")
#             p_val = float(p_str)
#             per_p_files.append((p_val, os.path.join(output_dir, fname)))
#         except ValueError:
#             # Skip files that don't contain a float p
#             pass
#
#
# per_p_files.sort(key=lambda x: x[0])  # ascending by p
# print('Consolidating files for p-values:', [p for p, _ in per_p_files])
#
# consolidated_df = None
# for _, fpath in per_p_files:
#     try:
#         part_df = pd.read_hdf(fpath, key="df")
#     except (OSError, KeyError):
#         continue
#     if consolidated_df is None or consolidated_df.empty:
#         consolidated_df = part_df
#     else:
#         consolidated_df = pd.concat([consolidated_df, part_df], ignore_index=True)
#
# if consolidated_df is not None and not consolidated_df.empty:
#     consolidated_df.to_hdf(consolidated_filename, mode="w", key="df")
