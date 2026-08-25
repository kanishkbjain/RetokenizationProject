import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.gsm8k_python.data import (  # noqa: E402
    DATA_PATH as GSM8K_JSON_PATH,
    build_problem_records,
    extract_code_from_generation,
    write_problem_file,
)
from eval.humaneval.evaluation import evaluate_functional_correctness  # noqa: E402
from eval.util import load_model_and_tokenizer, load_tokenizer  # noqa: E402
from olmo.torch_util import seed_all  # noqa: E402


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
        "Run GSM8K-Python temperature sampling from already materialized retokenized prompts. "
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
    default="allenai/OLMo-2-1124-7B",
)
parser.add_argument("--step", type=int, default=None)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Full checkpoint name/revision, e.g. 'stage1-step928646-tokens3896B'.",
)
parser.add_argument("--max_num_examples", type=int, default=1000)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--pass_at_k", type=int, default=10)
parser.add_argument("--unbiased_sampling_size_n", type=int, default=11)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature used for decoding from each saved retokenized prompt.",
)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--max_new_tokens", type=int, default=500)
parser.add_argument(
    "--source_temp_ident",
    type=str,
    default="",
    help="Temp identifier suffix on the completed source retok run.",
)
parser.add_argument(
    "--output_temp_ident",
    type=str,
    default="_temp_from_retok",
    help="Suffix for the new combined retok+temperature output dirs.",
)
parser.add_argument(
    "--source_sample_p0",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="For p=0.0 only, read the sampled p=0 source dir instead of the dontsample source dir.",
)
parser.add_argument(
    "--overwrite_samples",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Overwrite existing predictions/scored outputs.",
)
parser.add_argument(
    "--dry_run",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Only validate source rows and recovered prompt token prefixes; do not load the model or generate.",
)

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


seed_all(42)

GSM8K_DIR = PROJECT_ROOT / "eval" / "gsm8k"
BASE_OUTPUT_DIR = Path("/scratch/kjain25/Tokenizer_passK/results_passretok/gsm8k_python")


def parse_p_values(parsed_args):
    if parsed_args.pvals:
        return [float(x) for x in parsed_args.pvals.split(",") if x.strip() != ""]
    if parsed_args.p is not None:
        return [float(parsed_args.p)]
    return [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]


def model_output_dir_name(model_name, step=None, checkpoint=None):
    model_dir_name = model_name.replace("/", "_")
    if step is not None and checkpoint is not None:
        raise ValueError("Specify only one of `--step` or `--checkpoint`.")
    if checkpoint is not None:
        return f"{model_dir_name}_{checkpoint}"
    if step is None:
        return model_dir_name
    return f"{model_dir_name}_step_{step}"


def load_base_test_df(max_num_examples):
    if max_num_examples:
        sampled_path = GSM8K_DIR / f"sampled_data_{max_num_examples}.h5"
        if sampled_path.exists():
            print("Loading existing sampled GSM8K test data.")
            base_examples_df = pd.read_hdf(sampled_path, key="df")
        else:
            base_examples_df = pd.read_json(GSM8K_JSON_PATH, lines=True)
            base_examples_df = base_examples_df.sample(min(len(base_examples_df), max_num_examples), random_state=42)
            base_examples_df = base_examples_df.reset_index(drop=True)
            base_examples_df["temp_len"] = base_examples_df["question"].apply(len)
            base_examples_df.sort_values("temp_len", inplace=True, ascending=False)
            base_examples_df.drop(columns="temp_len", inplace=True)
            base_examples_df = base_examples_df.reset_index(drop=True)
            base_examples_df.to_hdf(sampled_path, key="df", mode="w")
            print("Sampled and saved new GSM8K test data.")
    else:
        base_examples_df = pd.read_json(GSM8K_JSON_PATH, lines=True)

    problem_records = build_problem_records(base_examples_df.to_dict(orient="records"))
    test_df = pd.DataFrame(problem_records)
    test_df["temp_len"] = test_df["prompt"].apply(len)
    test_df.sort_values("temp_len", inplace=True, ascending=False)
    test_df.drop(columns="temp_len", inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return test_df, len(test_df)


def expected_rows_for_p(retokenizationp, max_num_examples, unbiased_sampling_size_n, source_sample_p0):
    if retokenizationp == 0.0:
        return max_num_examples * (unbiased_sampling_size_n * 5 if source_sample_p0 else 1)
    return max_num_examples * unbiased_sampling_size_n


def run_output_dir(output_dir, retokenizationp, max_num_examples, unbiased_sampling_size_n, temp_ident, p0_dontsample):
    run_dir = output_dir / (
        f"retokp_{retokenizationp}_maxexamples_{max_num_examples}_unbiasedsize_{unbiased_sampling_size_n}{temp_ident}"
    )
    if retokenizationp == 0.0 and p0_dontsample:
        run_dir = run_dir / "dontsample"
    return run_dir


def source_predictions_path(source_run_dir):
    scored_path = source_run_dir / "scored_predictions.jsonl"
    predictions_path = source_run_dir / "predictions.jsonl"
    if scored_path.exists():
        return scored_path
    if predictions_path.exists():
        return predictions_path
    raise FileNotFoundError(
        "Could not find a completed source predictions file.\n"
        f"Expected one of:\n  - {scored_path}\n  - {predictions_path}"
    )


def load_source_predictions(source_run_dir, expected_rows):
    path = source_predictions_path(source_run_dir)
    df = pd.read_json(path, lines=True)
    if len(df) != expected_rows:
        raise ValueError(f"Source predictions at {path} have {len(df)} rows; expected {expected_rows}.")
    required_columns = {"task_id", "prompt", "generation_tokens"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Source predictions at {path} are missing columns: {missing_columns}")
    return path, df


def _strip_leading_special_tokens(token_ids, tokenizer):
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    special_prefix_ids = {token_id for token_id in (pad_token_id, bos_token_id) if token_id is not None}
    start = 0
    while start < len(token_ids) and int(token_ids[start]) in special_prefix_ids:
        start += 1
    return [int(token_id) for token_id in token_ids[start:]]


def recover_prompt_token_ids(row, tokenizer):
    prompt = row["prompt"]
    token_ids = _strip_leading_special_tokens(row["generation_tokens"], tokenizer)
    if not token_ids:
        raise ValueError(f"Empty generation_tokens for task_id={row['task_id']}.")

    for end in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:end], skip_special_tokens=True)
        if decoded == prompt:
            return token_ids[:end]

    preview = tokenizer.decode(token_ids[: min(len(token_ids), 128)], skip_special_tokens=True)
    raise ValueError(
        f"Could not recover retokenized prompt prefix for task_id={row['task_id']}.\n"
        f"Prompt starts with: {prompt[:200]!r}\n"
        f"Decoded token prefix starts with: {preview[:200]!r}"
    )


def recover_all_prompt_token_ids(source_df, tokenizer):
    prompt_token_ids = []
    records = source_df.to_dict(orient="records")
    for row in tqdm(records, desc="Recovering retokenized prompt tokens"):
        prompt_token_ids.append(recover_prompt_token_ids(row, tokenizer))
    return prompt_token_ids


def load_complete_predictions(predictions_path, expected_rows):
    if not predictions_path.exists():
        return None
    try:
        predictions = pd.read_json(predictions_path, lines=True)
    except (OSError, ValueError) as exc:
        print(f"Could not read predictions at {predictions_path}: {exc}. Regenerating.")
        return None
    if len(predictions) != expected_rows:
        print(
            f"Found incomplete predictions at {predictions_path} "
            f"({len(predictions)} / {expected_rows} rows). Regenerating."
        )
        return None
    return predictions


def pad_prompt_batch(prompt_token_ids, tokenizer):
    padded = tokenizer.pad(
        {"input_ids": prompt_token_ids},
        padding="longest",
        padding_side="left",
        return_tensors="pt",
    )
    attention_mask = padded["input_ids"].ne(tokenizer.pad_token_id).long()
    padded["attention_mask"] = attention_mask
    return padded


def load_cached_generations(generation_cache_dir, total_rows):
    generations = [None] * total_rows
    generation_tokens = [None] * total_rows
    if generation_cache_dir is None or not generation_cache_dir.exists():
        return generations, generation_tokens

    for path in sorted(generation_cache_dir.glob("generated_sequences_batch_*.jsonl")):
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
                    prompt_idx = record.get("prompt_idx")
                    if not isinstance(prompt_idx, int) or prompt_idx < 0 or prompt_idx >= total_rows:
                        continue
                    generation = record.get("generation")
                    tokens = record.get("generation_tokens")
                    if generation is None or tokens is None:
                        continue
                    generations[prompt_idx] = generation
                    generation_tokens[prompt_idx] = tokens
        except OSError:
            continue
    return generations, generation_tokens


def next_cache_batch_id(generation_cache_dir):
    max_id = -1
    try:
        for path in generation_cache_dir.glob("generated_sequences_batch_*.jsonl"):
            core = path.name[len("generated_sequences_batch_") : -len(".jsonl")]
            if core.isdigit():
                max_id = max(max_id, int(core))
    except OSError:
        pass
    return max_id + 1


def generate_from_prompt_tokens(
    model,
    tokenizer,
    source_df,
    prompt_token_ids,
    batch_size,
    temperature,
    top_p,
    max_new_tokens,
    generation_cache_dir,
):
    rows = source_df.to_dict(orient="records")
    outputs, generation_tokens = load_cached_generations(generation_cache_dir, len(rows))
    pending_indices = [idx for idx, output in enumerate(outputs) if output is None]
    if generation_cache_dir is not None:
        os.makedirs(generation_cache_dir, exist_ok=True)
    next_batch_id = next_cache_batch_id(generation_cache_dir)

    if len(pending_indices) < len(rows):
        print(f"Resuming generation: {len(pending_indices)} / {len(rows)} prompts remaining.")

    generation_kwargs = {
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }
    print("Using sampling generation kwargs for recovered retokenized prompts", generation_kwargs)

    for start in tqdm(range(0, len(pending_indices), batch_size), desc="Generating"):
        batch_prompt_indices = pending_indices[start : start + batch_size]
        batch_prompt_ids = [prompt_token_ids[idx] for idx in batch_prompt_indices]
        batch_inputs = pad_prompt_batch(batch_prompt_ids, tokenizer)
        batch_inputs = {
            key: (value.to(model.device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch_inputs.items()
        }
        with torch.inference_mode():
            batch_outputs = model.generate(
                **batch_inputs,
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
            outputs[prompt_idx] = extract_code_from_generation(batch_generations[local_idx])
            generation_tokens[prompt_idx] = batch_sequences[local_idx]

        if generation_cache_dir is not None:
            batch_seq_path = generation_cache_dir / f"generated_sequences_batch_{next_batch_id:06d}.jsonl"
            next_batch_id += 1
            with open(batch_seq_path, "w") as fo:
                for local_idx, prompt_idx in enumerate(batch_prompt_indices):
                    record = {
                        "prompt_idx": prompt_idx,
                        "generation": outputs[prompt_idx],
                        "generation_tokens": generation_tokens[prompt_idx],
                    }
                    fo.write(json.dumps(record) + "\n")

        del batch_inputs, batch_outputs
        gc.collect()
        torch.cuda.empty_cache()

    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "completion": out,
            "generation_tokens": tokens,
            "source_prompt_tokens": source_prompt_tokens,
        }
        for row, out, tokens, source_prompt_tokens in zip(
            rows,
            outputs,
            generation_tokens,
            prompt_token_ids,
        )
    ]


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}.")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError(f"--top_p must be in (0, 1], got {args.top_p}.")
    if args.step is not None and args.checkpoint is not None:
        raise ValueError("Specify only one of `--step` or `--checkpoint`.")

    p_vals = parse_p_values(args)
    model_name = args.model_name_or_path
    model_dir = model_output_dir_name(model_name, step=args.step, checkpoint=args.checkpoint)
    experiment_output_dir = BASE_OUTPUT_DIR / model_dir
    print(experiment_output_dir)

    base_test_df, max_num_examples = load_base_test_df(args.max_num_examples)

    if args.dry_run:
        model = None
        tokenizer = load_tokenizer(model_name)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, step=args.step, checkpoint=args.checkpoint)

    for retokenizationp in p_vals:
        expected_rows = expected_rows_for_p(
            retokenizationp=retokenizationp,
            max_num_examples=max_num_examples,
            unbiased_sampling_size_n=args.unbiased_sampling_size_n,
            source_sample_p0=args.source_sample_p0,
        )
        source_run_dir = run_output_dir(
            output_dir=experiment_output_dir,
            retokenizationp=retokenizationp,
            max_num_examples=max_num_examples,
            unbiased_sampling_size_n=args.unbiased_sampling_size_n,
            temp_ident=args.source_temp_ident,
            p0_dontsample=(retokenizationp == 0.0 and not args.source_sample_p0),
        )
        combined_run_dir = run_output_dir(
            output_dir=experiment_output_dir,
            retokenizationp=retokenizationp,
            max_num_examples=max_num_examples,
            unbiased_sampling_size_n=args.unbiased_sampling_size_n,
            temp_ident=args.output_temp_ident,
            p0_dontsample=False,
        )
        if source_run_dir.resolve() == combined_run_dir.resolve():
            raise ValueError(f"Source and output directories are the same: {source_run_dir}")

        source_path, source_df = load_source_predictions(source_run_dir, expected_rows)
        print(f"Loaded source rows from {source_path}")
        prompt_token_ids = recover_all_prompt_token_ids(source_df, tokenizer)
        print(f"Recovered {len(prompt_token_ids)} retokenized prompt token sequences for p={retokenizationp}.")

        if args.dry_run:
            continue

        predictions_path = combined_run_dir / "predictions.jsonl"
        scored_predictions_path = combined_run_dir / "scored_predictions.jsonl"
        generation_cache_dir = combined_run_dir / "generation_cache"
        problem_file = combined_run_dir / "problems.jsonl"

        if scored_predictions_path.exists() and not args.overwrite_samples:
            print(f"Found existing scored predictions at {scored_predictions_path}, skipping.")
            continue

        os.makedirs(combined_run_dir, exist_ok=True)
        write_problem_file(base_test_df.to_dict(orient="records"), problem_file)

        predictions = None
        if not args.overwrite_samples:
            predictions = load_complete_predictions(predictions_path, expected_rows)
        if predictions is not None:
            print(f"Found existing predictions at {predictions_path}, skipping generation.")
        else:
            predictions = generate_from_prompt_tokens(
                model=model,
                tokenizer=tokenizer,
                source_df=source_df,
                prompt_token_ids=prompt_token_ids,
                batch_size=args.eval_batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                generation_cache_dir=generation_cache_dir,
            )
            predictions = pd.DataFrame(predictions)
            predictions.to_json(predictions_path, orient="records", lines=True)

        metrics = evaluate_functional_correctness(
            sample_file=str(predictions_path),
            k=[args.pass_at_k],
            n_workers=64,
            problem_file=str(problem_file),
        )
        metrics["num_examples"] = len(base_test_df)
        metrics["source_predictions_path"] = str(source_path)
        metrics["temperature"] = args.temperature
        metrics["top_p"] = args.top_p
        for key, value in metrics.items():
            print(f"{key}: {value}")

        results = pd.read_json(scored_predictions_path, lines=True)
        with open(combined_run_dir / "example_prompt.txt", "w") as fo:
            fo.write(results.iloc[0]["prompt"])
        with open(combined_run_dir / "metrics.json", "w") as fo:
            json.dump(metrics, fo, indent=4)


if __name__ == "__main__":
    main()
