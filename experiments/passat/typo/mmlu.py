import argparse
import contextlib
import gc
import os
import re
from pathlib import Path

import pandas as pd
import torch
import tqdm

from eval import util
from eval.paths import get_results_dataset_dir


parser = argparse.ArgumentParser(description="Run passattypos on MMLU with configurable GPU(s) and p-value(s).")
parser.add_argument("--gpus", type=str, help="GPU id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--p", type=float, help="Single typo probability, e.g. 0.4.")
parser.add_argument("--pvals", type=str, help="Comma-separated typo probabilities, e.g. '0.0,0.2,0.4'.")
parser.add_argument("--temp_ident", type=str, help="Temporary identifier to add.", default="")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Model name or path.",
    default="allenai/OLMo-2-1124-7B-Instruct",
)
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


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


output_dir = str(get_results_dataset_dir("typo", "mmlu"))
model_name = args.model_name_or_path
reasoning = False
dataset = "mmlu"
N = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "eval" / dataset / f"sampled_data_{N}_sorted.h5"
RAW_DATA_PATH = PROJECT_ROOT / "eval" / dataset / "test.jsonl"

no_prefix_suffix = False
answer_prefix = "Answer and Reasoning:" if reasoning else "Answer:"
max_new_tokens = 512 if reasoning else 1

if args.pvals:
    p_vals = [float(x) for x in args.pvals.split(",") if x.strip() != ""]
elif args.p is not None:
    p_vals = [float(args.p)]
else:
    p_vals = [0.00, 0.20, 0.40, 0.60, 0.80, 1.0]

n_typo_variants = 11
n_baseline_variants = 1

if DATA_PATH.exists():
    test_df = pd.read_hdf(DATA_PATH, key="df")
else:
    test_df = pd.read_json(RAW_DATA_PATH, lines=True)
    test_df = test_df.sample(min(len(test_df), N), random_state=42).reset_index(drop=True)
    test_df["temp_len"] = test_df["question"].apply(len)
    test_df.sort_values("temp_len", inplace=True, ascending=False)
    test_df.drop(columns="temp_len", inplace=True)
    test_df.to_hdf(DATA_PATH, key="df", mode="w")
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
                qa_format="False",
            )
            + "\n\n"
        )
    prompt += util.format_example(
        row["question"].strip(),
        choices=row["choices"],
        qa_format="False",
        answer_prefix=answer_prefix,
    )
    for _ in range(n_typo_variants):
        prompts.append(prompt)
        prompts_i.append(i)
        answers.append("ABCD"[row["answer"]])
        task_ids.append(f"mmlu/{i}")
    for _ in range(n_baseline_variants):
        prompts0.append(prompt)
        prompts_i_0.append(i)
        answers_0.append("ABCD"[row["answer"]])
        task_ids_0.append(f"mmlu/{i}")

batch_size = {3: 16, 50: 32, 200: 64, 400: 128}
chkpoint = 1000

if reasoning:
    output_dir = f"{output_dir}/reasoning"

output_dir = f"{output_dir}/{model_name.replace('/', '_')}/"
base_prefix = f"{output_dir}/{dataset}_N_{N}_numvariants_{n_typo_variants}"
os.makedirs(output_dir, exist_ok=True)

df_data = None
filename = None
data = []


def flush_data_chunk():
    global df_data, data, filename
    if not data:
        return
    chunk_df_data = pd.DataFrame.from_records(data)
    if df_data is None or df_data.empty:
        df_data = chunk_df_data
    else:
        df_data = pd.concat([df_data, chunk_df_data], ignore_index=True)
    df_data.to_hdf(filename, mode="w", key="df")
    data.clear()


def select_batch_size(prompti):
    for thresh in sorted(batch_size.keys()):
        if prompti < thresh:
            return batch_size[thresh]
    return batch_size[sorted(batch_size.keys())[-1]]


gc.collect()
torch.cuda.empty_cache()

for typo_p in p_vals[::-1]:
    filename = f"{base_prefix}_typop_{typo_p:.2f}{args.temp_ident}.df"

    df_data = pd.read_hdf(filename, key="df") if os.path.exists(filename) else None
    data.clear()

    if df_data is not None and not df_data.empty:
        start_prompti = df_data.shape[0]
        print("Resuming from prompt index", start_prompti, "for p=", typo_p)
    else:
        start_prompti = 0

    new_added = 0
    total_prompts = len(prompts0) if typo_p == 0.0 else len(prompts)
    prompti = start_prompti
    with tqdm.tqdm(total=total_prompts - start_prompti, desc=f"Processing p={typo_p:0.2f}") as pbar:
        while prompti < total_prompts:
            if new_added >= chkpoint:
                flush_data_chunk()
                new_added = 0

            prompti_ = prompts_i_0[prompti] if typo_p == 0.0 else prompts_i[prompti]
            bs = select_batch_size(prompti_)
            pbar.set_description(f"Processing p={typo_p:0.2f}, bs={bs}")

            if typo_p == 0.0:
                generation_kwargs = {
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "tokenizer": tokenizer,
                }
                batch_prompts = prompts0[prompti : prompti + bs]
                batch_prompts_i = prompts_i_0[prompti : prompti + bs]
                batch_answers = answers_0[prompti : prompti + bs]
                batch_task_ids = task_ids_0[prompti : prompti + bs]
            else:
                generation_kwargs = {
                    "num_return_sequences": 1,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "tokenizer": tokenizer,
                }
                batch_prompts = prompts[prompti : prompti + bs]
                batch_prompts_i = prompts_i[prompti : prompti + bs]
                batch_answers = answers[prompti : prompti + bs]
                batch_task_ids = task_ids[prompti : prompti + bs]

            if not batch_prompts:
                break

            batch_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                add_special_tokens=True,
                padding="longest",
            )

            res, batch_token_lengths = util.fiddle_tokens_typos(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                tokenizer,
                no_prefix_suffix=no_prefix_suffix,
                typo_p=typo_p,
                is_mcq=True,
                reason=reasoning,
            )
            if prompti == start_prompti:
                print("Sample perturbed input:", tokenizer.decode(res["input_ids"][0], skip_special_tokens=True))

            use_autocast = model.device.type == "cuda"
            res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in res.items()}
            with torch.inference_mode():
                with (
                    torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    if use_autocast
                    else contextlib.nullcontext()
                ):
                    logits_prompt = model(**res, return_dict=True).logits
                    probs_prompt = torch.softmax(logits_prompt[:, -1, :], dim=-1)
                    answer_idx = torch.tensor(
                        [tokenizer.convert_tokens_to_ids(answer) for answer in batch_answers],
                        device=probs_prompt.device,
                    ).unsqueeze(-1)
                    answer_probs = probs_prompt.gather(-1, answer_idx).squeeze(-1).cpu().tolist()
                    batch_outputs = model.generate(**res, **generation_kwargs)
                    len_inputs = res["input_ids"].shape[1]
                    for bi in range(len(batch_prompts)):
                        generated_token_ids = batch_outputs.sequences[bi][len_inputs:].tolist()
                        decoded_answer = decode_generated_answer(
                            generated_token_ids,
                            tokenizer=tokenizer,
                            reasoning=reasoning,
                        )
                        data.append(
                            {
                                "task_id": batch_task_ids[bi],
                                "p": typo_p,
                                "prompti": batch_prompts_i[bi],
                                "prompt_tokens": batch_outputs.sequences[bi].tolist()[:len_inputs],
                                "generated_tokens": generated_token_ids,
                                "token_lengths": batch_token_lengths[bi],
                                "answer": batch_answers[bi],
                                "answer_prob": answer_probs[bi],
                                "generated_answer": decoded_answer,
                                "passed": int(decoded_answer == batch_answers[bi]),
                            }
                        )

            del res, batch_outputs
            gc.collect()
            torch.cuda.empty_cache()
            step = len(batch_prompts)
            new_added += step
            prompti += step
            pbar.update(step)

    flush_data_chunk()
