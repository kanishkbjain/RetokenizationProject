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


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")

parser = argparse.ArgumentParser(description="Run passattypos on GSM8K with configurable GPU(s) and p-value(s).")
parser.add_argument("--gpus", type=str, help="GPU id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--p", type=float, help="Single typo probability, e.g. 0.4.")
parser.add_argument("--pvals", type=str, help="Comma-separated typo probabilities, e.g. '0.0,0.2,0.4'.")
parser.add_argument(
    "--sample",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Sample p=0.0 runs. Accepts either '--sample' or '--sample true/false'.",
)
parser.add_argument("--temp_ident", type=str, help="Temp identifier for output files.", default="")
parser.add_argument("--model_name_or_path", type=str, help="Model name or path.", default="allenai/OLMo-2-1124-7B-Instruct")
args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def format_answer(answer):
    answer = re.sub(r"<<.*?>>", "", answer)
    final_answer = answer.split("####")[-1].strip()
    sentences = answer.split("####")[0].strip().split("\n")
    sentences = [s + "." if not s.endswith(".") else s for s in sentences]
    return " ".join(sentences) + f"\n#### {final_answer}"


output_dir = str(get_results_dataset_dir("typo", "gsm8k"))
model_name = args.model_name_or_path
output_dir = f"{output_dir}/{model_name.replace('/', '_')}/"
dataset = "gsm8k"
N = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "eval" / dataset / f"sampled_data_{N}.h5"
RAW_DATA_PATH = PROJECT_ROOT / "eval" / dataset / "full_gsm8k_test.jsonl"

if args.pvals:
    p_vals = [float(x) for x in args.pvals.split(",") if x.strip() != ""]
elif args.p is not None:
    p_vals = [float(args.p)]
else:
    p_vals = [0.20, 0.40, 0.60, 0.80, 1.0]

n_typo_variants = 11
n_baseline_variants = 55 if args.sample else 1

if args.sample:
    print("Sampling enabled for p=0.0 runs.")

if DATA_PATH.exists():
    test_df = pd.read_hdf(DATA_PATH, key="df")
else:
    test_df = pd.read_json(RAW_DATA_PATH, lines=True)
    test_df = test_df.sample(min(len(test_df), N), random_state=42).reset_index(drop=True)
    test_df.to_hdf(DATA_PATH, key="df", mode="w")
test_df = test_df.reset_index(drop=True)
model, tokenizer = util.load_model_and_tokenizer(model_name)

num_incontext_examples = 0
qa_format = "qnan"
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
        prompt += util.format_example(ic_row["question"], answer=format_answer(ic_row["answer"]), qa_format=qa_format)
        prompt += "\n\n"
    prompt += util.format_example(row["question"], qa_format=qa_format)
    for _ in range(n_typo_variants):
        prompts.append(prompt)
        prompts_i.append(i)
        answers.append(float(row["answer"].split("#### ")[1].replace(",", "")))
        task_ids.append(f"gsm8k/{i}")
    for _ in range(n_baseline_variants):
        prompts0.append(prompt)
        prompts_i_0.append(i)
        answers_0.append(float(row["answer"].split("#### ")[1].replace(",", "")))
        task_ids_0.append(f"gsm8k/{i}")

batch_size = 64
chkpoint = 1000

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


gc.collect()
torch.cuda.empty_cache()

for typo_p in p_vals:
    filename = f"{base_prefix}_typop_{typo_p:.2f}{args.temp_ident}.df"
    if typo_p == 0.0:
        filename = f"{filename[:-3]}_sampled.df" if args.sample else filename
    df_data = pd.read_hdf(filename, key="df") if os.path.exists(filename) else None
    data.clear()

    if df_data is not None and not df_data.empty:
        start_prompti = df_data.shape[0]
        print("Resuming from prompt index", start_prompti, "for p=", typo_p)
    else:
        start_prompti = 0

    new_added = 0
    total_prompts = len(prompts0) if typo_p == 0.0 else len(prompts)
    for prompti in tqdm.tqdm(range(start_prompti, total_prompts, batch_size), desc=f"Processing p={typo_p:0.2f}"):
        if new_added >= chkpoint:
            flush_data_chunk()
            new_added = 0

        if typo_p == 0.0:
            generation_kwargs = {
                "num_return_sequences": 1,
                "max_new_tokens": 512,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.pad_token_id,
                "tokenizer": tokenizer,
                "do_sample": args.sample,
                "top_p": 0.9,
                "temperature": 1.0,
            }
            batch_prompts = prompts0[prompti : prompti + batch_size]
            batch_prompts_i = prompts_i_0[prompti : prompti + batch_size]
            batch_answers = answers_0[prompti : prompti + batch_size]
            batch_task_ids = task_ids_0[prompti : prompti + batch_size]
        else:
            generation_kwargs = {
                "num_return_sequences": 1,
                "max_new_tokens": 512,
                "do_sample": False,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.pad_token_id,
                "tokenizer": tokenizer,
            }
            batch_prompts = prompts[prompti : prompti + batch_size]
            batch_prompts_i = prompts_i[prompti : prompti + batch_size]
            batch_answers = answers[prompti : prompti + batch_size]
            batch_task_ids = task_ids[prompti : prompti + batch_size]

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
            is_mcq=False,
            reason=False,
            typo_p=typo_p,
        )
        use_autocast = model.device.type == "cuda"
        res = {k: (v.to(model.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in res.items()}
        with torch.inference_mode():
            with (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            ):
                batch_outputs = model.generate(**res, **generation_kwargs)
                len_inputs = res["input_ids"].shape[1]
                for bi in range(len(batch_prompts)):
                    decoded_answer = tokenizer.decode(batch_outputs.sequences[bi][len_inputs:], skip_special_tokens=True)
                    decoded_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", decoded_answer)
                    generated_answer = float(decoded_numbers[-1]) if decoded_numbers else None
                    data.append(
                        {
                            "task_id": batch_task_ids[bi],
                            "p": typo_p,
                            "prompti": batch_prompts_i[bi],
                            "prompt_tokens": batch_outputs.sequences[bi].tolist()[:len_inputs],
                            "generated_tokens": batch_outputs.sequences[bi].tolist()[len_inputs:],
                            "token_lengths": batch_token_lengths[bi],
                            "answer": batch_answers[bi],
                            "generated_answer": generated_answer,
                            "passed": int(generated_answer == batch_answers[bi]),
                        }
                    )

        del res, batch_outputs
        gc.collect()
        torch.cuda.empty_cache()
        new_added += len(batch_prompts)

    flush_data_chunk()
