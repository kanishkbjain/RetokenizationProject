import argparse
import contextlib
import gc
import json
import os
import re
import sys
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


parser = argparse.ArgumentParser(
    description=(
        "Run GSM8K temperature sampling from already materialized retokenized prompts. "
        "This combines retokenization prompt sampling with temperature decoding without regenerating retokenizations."
    )
)
parser.add_argument("--gpus", type=str, help="GPU id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--p", type=float, help="Single retokenization probability, e.g. 0.4.")
parser.add_argument("--pvals", type=str, help="Comma-separated retokenization probabilities, e.g. '0.0,0.2,0.4'.")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Model name or path.",
    default="allenai/OLMo-2-1124-7B-Instruct",
)
parser.add_argument("--N", type=int, default=1000, help="Number of GSM8K examples in the completed retok run.")
parser.add_argument("--n_retokenizations", type=int, default=11)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument(
    "--source_temp_ident",
    type=str,
    default="",
    help="Suffix on the completed source retok .df files.",
)
parser.add_argument(
    "--output_temp_ident",
    type=str,
    default="_temp_from_retok",
    help="Suffix for the new combined retok+temperature .df files.",
)
parser.add_argument(
    "--source_sample_p0",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="For p=0.0 only, read the sampled p=0 source .df instead of the greedy p=0 source .df.",
)
parser.add_argument(
    "--output_sample_p0",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="For p=0.0 only, write the output with the _sampled.df suffix.",
)
parser.add_argument(
    "--overwrite",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Regenerate from the beginning even if an output .df already exists.",
)
parser.add_argument(
    "--dry_run",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Validate source files and row counts without loading the model or generating.",
)

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd
import torch
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import util  # noqa: E402
from olmo.torch_util import seed_all  # noqa: E402


BASE_OUTPUT_DIR = Path("/scratch/kjain25/Tokenizer_passK/results_passretok/gsm8k")
DATASET = "gsm8k"


def parse_p_values(parsed_args):
    if parsed_args.pvals:
        return [float(x) for x in parsed_args.pvals.split(",") if x.strip()]
    if parsed_args.p is not None:
        return [float(parsed_args.p)]
    return [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]


def model_output_dir(model_name):
    return BASE_OUTPUT_DIR / model_name.replace("/", "_")


def retok_file_path(output_dir, p_value, n_examples, n_retokenizations, temp_ident, sampled_p0=False):
    filename = f"{DATASET}_N_{n_examples}_numretokenizations_{n_retokenizations}_{p_value:.2f}{temp_ident}.df"
    if p_value == 0.0 and sampled_p0:
        filename = filename[:-3] + "_sampled.df"
    return output_dir / filename


def expected_rows_for_p(p_value, n_examples, n_retokenizations, sampled_p0):
    if p_value == 0.0:
        return n_examples * (n_retokenizations * 5 if sampled_p0 else 1)
    return n_examples * n_retokenizations


def load_source_df(path, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"Could not find completed source retok file:\n  {path}")

    df = pd.read_hdf(path, key="df")
    if len(df) != expected_rows:
        raise ValueError(f"Source file has {len(df)} rows but expected {expected_rows}:\n  {path}")

    required_columns = {
        "p",
        "prompti",
        "prompt_tokens",
        "token_lengths",
        "answer",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Source file is missing required columns {missing_columns}:\n  {path}")

    if "task_id" not in df.columns:
        df = df.copy()
        df["task_id"] = df["prompti"].map(lambda prompti: f"{DATASET}/{int(prompti)}")

    return df


def load_existing_output(path, overwrite):
    if overwrite or not path.exists():
        return None
    try:
        return pd.read_hdf(path, key="df")
    except (OSError, KeyError, ValueError) as exc:
        print(f"Could not read existing output at {path}: {exc}. Regenerating.")
        return None


def generation_cache_dir_for_output(output_path):
    return output_path.parent / f"{output_path.stem}_generation_cache"


def pad_prompt_batch(prompt_token_ids, tokenizer):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        padded = tokenizer.pad(
            {"input_ids": prompt_token_ids},
            padding="longest",
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side

    attention_mask = padded["input_ids"].ne(tokenizer.pad_token_id).long()
    padded["attention_mask"] = attention_mask
    return padded


def decode_gsm8k_answer(decoded_text):
    decoded_answer = re.findall(r"[-+]?\d*\.\d+|\d+", decoded_text)
    return float(decoded_answer[-1]) if len(decoded_answer) > 0 else None


def load_cached_rows(generation_cache_dir, total_rows):
    cached_rows = {}
    if not generation_cache_dir.exists():
        return cached_rows

    for path in sorted(generation_cache_dir.glob("generated_rows_batch_*.jsonl")):
        try:
            with open(path, "r") as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    source_idx = record.get("source_idx")
                    if not isinstance(source_idx, int) or source_idx < 0 or source_idx >= total_rows:
                        continue
                    row = record.get("row")
                    if not isinstance(row, dict):
                        continue
                    cached_rows[source_idx] = row
        except OSError:
            continue
    return cached_rows


def seed_cached_rows_from_existing_output(cached_rows, existing_df, total_rows):
    if existing_df is None or existing_df.empty:
        return cached_rows

    for source_idx, row in enumerate(existing_df.to_dict(orient="records")):
        if source_idx >= total_rows:
            break
        cached_rows.setdefault(source_idx, row)
    return cached_rows


def next_cache_batch_id(generation_cache_dir):
    max_id = -1
    try:
        for path in generation_cache_dir.glob("generated_rows_batch_*.jsonl"):
            core = path.name[len("generated_rows_batch_") : -len(".jsonl")]
            if core.isdigit():
                max_id = max(max_id, int(core))
    except OSError:
        pass
    return max_id + 1


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_cache_batch(generation_cache_dir, batch_id, batch_records):
    os.makedirs(generation_cache_dir, exist_ok=True)
    cache_path = generation_cache_dir / f"generated_rows_batch_{batch_id:06d}.jsonl"
    with open(cache_path, "w") as fo:
        for source_idx, row in batch_records:
            fo.write(json.dumps({"source_idx": source_idx, "row": row}, default=json_default) + "\n")


def write_output_from_cache(output_path, cached_rows):
    if not cached_rows:
        return None

    ordered_indices = sorted(cached_rows)
    df = pd.DataFrame.from_records([cached_rows[idx] for idx in ordered_indices])
    df.to_hdf(output_path, mode="w", key="df")
    return df


def generate_from_source_df(model, tokenizer, source_df, output_path, existing_df):
    total_rows = source_df.shape[0]
    if existing_df is not None and existing_df.shape[0] > total_rows:
        raise ValueError(f"Existing output has more rows than the source file:\n  {output_path}")

    generation_cache_dir = generation_cache_dir_for_output(output_path)
    cached_rows = load_cached_rows(generation_cache_dir, total_rows)
    cached_rows = seed_cached_rows_from_existing_output(cached_rows, existing_df, total_rows)

    if len(cached_rows) == total_rows:
        write_output_from_cache(output_path, cached_rows)
        print(f"Found complete output at {output_path}, skipping.")
        return pd.DataFrame.from_records([cached_rows[idx] for idx in sorted(cached_rows)])

    if len(cached_rows) > 0:
        print(f"Resuming generation: {total_rows - len(cached_rows)} / {total_rows} rows remaining.")

    os.makedirs(output_path.parent, exist_ok=True)
    rows = source_df.to_dict(orient="records")
    chkpoint = 1000
    new_added = 0
    next_batch_id = next_cache_batch_id(generation_cache_dir)

    generation_kwargs = {
        "num_return_sequences": 1,
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizer": tokenizer,
        "do_sample": True,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    print("Using sampling generation kwargs for recovered retokenized prompts:", generation_kwargs)

    pending_indices = [idx for idx in range(total_rows) if idx not in cached_rows]
    for pending_start in tqdm.tqdm(
        range(0, len(pending_indices), args.eval_batch_size),
        desc=f"Processing {output_path.name}",
    ):
        if new_added >= chkpoint:
            existing_df = write_output_from_cache(output_path, cached_rows)
            new_added = 0

        batch_indices = pending_indices[pending_start : pending_start + args.eval_batch_size]
        batch_rows = [rows[idx] for idx in batch_indices]
        if len(batch_rows) == 0:
            break

        batch_prompt_tokens = [row["prompt_tokens"] for row in batch_rows]
        res = pad_prompt_batch(batch_prompt_tokens, tokenizer)

        use_autocast = model.device.type == "cuda"
        res = {
            key: (value.to(model.device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in res.items()
        }

        with torch.inference_mode():
            with (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            ):
                batch_outputs = model.generate(**res, **generation_kwargs)
                len_inputs = res["input_ids"].shape[1]
                batch_cache_records = []
                for bi, row in enumerate(batch_rows):
                    generated_tokens = batch_outputs.sequences[bi].tolist()[len_inputs:]
                    decoded_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    decoded_answer = decode_gsm8k_answer(decoded_answer)
                    passed = int(decoded_answer == row["answer"])
                    output_row = {
                        "task_id": row["task_id"],
                        "p": row["p"],
                        "prompti": row["prompti"],
                        "prompt_tokens": row["prompt_tokens"],
                        "generated_tokens": generated_tokens,
                        "token_lengths": row["token_lengths"],
                        "answer": row["answer"],
                        "generated_answer": decoded_answer,
                        "passed": passed,
                    }
                    source_idx = batch_indices[bi]
                    cached_rows[source_idx] = output_row
                    batch_cache_records.append((source_idx, output_row))

                write_cache_batch(generation_cache_dir, next_batch_id, batch_cache_records)
                next_batch_id += 1

        del res, batch_outputs
        gc.collect()
        torch.cuda.empty_cache()
        new_added += len(batch_rows)

    return write_output_from_cache(output_path, cached_rows)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}.")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"--top_p must be in (0, 1], got {args.top_p}.")

    seed_all(42)

    p_vals = parse_p_values(args)
    output_dir = model_output_dir(args.model_name_or_path)
    print(output_dir)

    if args.dry_run:
        model = None
        tokenizer = None
    else:
        model, tokenizer = util.load_model_and_tokenizer(args.model_name_or_path)

    for retokp in p_vals:
        source_sampled_p0 = retokp == 0.0 and args.source_sample_p0
        output_sampled_p0 = retokp == 0.0 and args.output_sample_p0
        expected_rows = expected_rows_for_p(
            p_value=retokp,
            n_examples=args.N,
            n_retokenizations=args.n_retokenizations,
            sampled_p0=source_sampled_p0,
        )

        source_path = retok_file_path(
            output_dir=output_dir,
            p_value=retokp,
            n_examples=args.N,
            n_retokenizations=args.n_retokenizations,
            temp_ident=args.source_temp_ident,
            sampled_p0=source_sampled_p0,
        )
        output_path = retok_file_path(
            output_dir=output_dir,
            p_value=retokp,
            n_examples=args.N,
            n_retokenizations=args.n_retokenizations,
            temp_ident=args.output_temp_ident,
            sampled_p0=output_sampled_p0,
        )

        if source_path.resolve() == output_path.resolve():
            raise ValueError(f"Source and output paths are the same:\n  {source_path}")

        source_df = load_source_df(source_path, expected_rows)
        print(f"Loaded {len(source_df)} source rows from {source_path}")

        if args.dry_run:
            print(f"Dry run: would write {output_path}")
            continue

        existing_df = load_existing_output(output_path, overwrite=args.overwrite)
        generate_from_source_df(
            model=model,
            tokenizer=tokenizer,
            source_df=source_df,
            output_path=output_path,
            existing_df=existing_df,
        )


if __name__ == "__main__":
    main()
