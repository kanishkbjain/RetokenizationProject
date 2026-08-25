import argparse
import json
import os
from pathlib import Path

import pandas as pd

from eval.gsm8k_python.data import (
    DATA_PATH as GSM8K_JSON_PATH,
    build_problem_records,
    extract_code_from_generation,
    write_problem_file,
)
from eval.humaneval.evaluation import evaluate_functional_correctness
from eval.paths import get_results_dataset_dir
from eval.util import batched_generate_legacy, load_model_and_tokenizer
from olmo.torch_util import seed_all


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


parser = argparse.ArgumentParser(description="Run passattypos GSM8K-Python with configurable GPU(s) and p-value(s).")
parser.add_argument("--gpus", type=str, help="GPU id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--p", type=float, help="Single typo probability, e.g. 0.4.")
parser.add_argument("--pvals", type=str, help="Comma-separated typo probabilities, e.g. '0.0,0.2,0.4'.")
parser.add_argument(
    "--sample",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Sample for p=0.0 runs. Accepts either '--sample' or '--sample true/false'.",
)
parser.add_argument("--temp_ident", type=str, help="Temp identifier for output dirs.", default="")
parser.add_argument("--model_name_or_path", type=str, help="Model name or path.", default="allenai/OLMo-2-1124-7B")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--max_num_examples", type=int, default=1000)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--pass_at_k", type=int, default=10)
parser.add_argument("--unbiased_sampling_size_n", type=int, default=11)
parser.add_argument(
    "--overwrite_samples",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Overwrite existing predictions/scored outputs.",
)
parser.add_argument(
    "--hidden_states",
    nargs="?",
    const=True,
    default=False,
    type=str2bool,
    help="Store generation hidden states.",
)

args = parser.parse_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

seed_all(42)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GSM8K_DIR = PROJECT_ROOT / "eval" / "gsm8k"
BASE_OUTPUT_DIR = get_results_dataset_dir("typo", "gsm8k_python")



def parse_p_values(parsed_args):
    if parsed_args.pvals:
        return [float(x) for x in parsed_args.pvals.split(",") if x.strip() != ""]
    if parsed_args.p is not None:
        return [float(parsed_args.p)]
    return [0.20, 0.40, 0.60, 0.80, 1.0]


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


def build_generation_frames(test_df, unbiased_sampling_size_n, sample_p0):
    if sample_p0:
        p0_df = pd.concat([test_df] * (unbiased_sampling_size_n * 5), ignore_index=True)
    else:
        p0_df = test_df.copy()
    typo_df = pd.concat([test_df] * unbiased_sampling_size_n, ignore_index=True)
    return p0_df, typo_df


def load_complete_predictions(predictions_path, expected_rows):
    if not predictions_path.exists():
        return None
    try:
        predictions = pd.read_json(predictions_path, lines=True)
    except (OSError, ValueError) as exc:
        print(f"Could not read predictions at {predictions_path}: {exc}. Rebuilding from generation cache.")
        return None
    if len(predictions) != expected_rows:
        print(
            f"Found incomplete predictions at {predictions_path} "
            f"({len(predictions)} / {expected_rows} rows). Rebuilding from generation cache."
        )
        return None
    return predictions


def evaluate_gsm8k_python(
    model,
    tokenizer,
    test_df,
    batch_size,
    typo_p,
    use_sampling=False,
    return_hidden_states=False,
    generation_cache_dir=None,
):
    prompts = test_df.prompt.tolist()

    print(f"--- GSM8K-Python example prompt ---\n{prompts[0]}\n----------------------")

    generation_kwargs = {"do_sample": False, "max_new_tokens": 500}
    if use_sampling:
        generation_kwargs = {"do_sample": True, "top_p": 0.9, "temperature": 1.0, "max_new_tokens": 500}
        print("Using sampling generation kwargs for typo_p=0.0", generation_kwargs)

    outputs_generated = batched_generate_legacy(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        perturbation_mode="typo",
        typo_p=typo_p,
        typo_region="docstring",
        return_last_hidden_states=return_hidden_states,
        hidden_states_dir=generation_cache_dir,
        store_hidden_states_in_memory=False,
        hidden_states_pre_norm=True,
        **generation_kwargs,
    )
    outputs = [extract_code_from_generation(out) for out in outputs_generated["generations"]]

    if return_hidden_states:
        return [
            {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "completion": out,
                "generation_tokens": tokens,
                "generation_hidden_state_hash": hidden_state_hash,
            }
            for ex, out, tokens, hidden_state_hash in zip(
                test_df.to_dict(orient="records"),
                outputs,
                outputs_generated["generation_tokens"],
                outputs_generated["generation_hidden_state_hashes"],
            )
        ]

    return [
        {
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "completion": out,
            "generation_tokens": tokens,
        }
        for ex, out, tokens in zip(
            test_df.to_dict(orient="records"),
            outputs,
            outputs_generated["generation_tokens"],
        )
    ]


def run_output_dir(output_dir, typo_p, max_num_examples, unbiased_sampling_size_n, sample_p0, temp_ident):
    run_dir = output_dir / f"typop_{typo_p}_maxexamples_{max_num_examples}_unbiasedsize_{unbiased_sampling_size_n}{temp_ident}"
    if typo_p == 0.0 and not sample_p0:
        run_dir = run_dir / "dontsample"
    return run_dir


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    p_vals = parse_p_values(args)
    model_name = args.model_name_or_path
    output_dir = BASE_OUTPUT_DIR / model_name.replace("/", "_")
    print(output_dir)

    model, tokenizer = load_model_and_tokenizer(model_name, step=args.step)
    base_test_df, max_num_examples = load_base_test_df(args.max_num_examples)
    p0_df, typo_df = build_generation_frames(
        base_test_df,
        unbiased_sampling_size_n=args.unbiased_sampling_size_n,
        sample_p0=args.sample,
    )

    os.makedirs(output_dir, exist_ok=True)

    for typo_p in p_vals:
        test_df = p0_df if typo_p == 0.0 else typo_df
        expected_rows = len(test_df)
        run_output_dir_ = run_output_dir(
            output_dir=output_dir,
            typo_p=typo_p,
            max_num_examples=max_num_examples,
            unbiased_sampling_size_n=args.unbiased_sampling_size_n,
            sample_p0=args.sample,
            temp_ident=args.temp_ident,
        )
        predictions_path = run_output_dir_ / "predictions.jsonl"
        scored_predictions_path = run_output_dir_ / "scored_predictions.jsonl"
        generation_cache_dir = run_output_dir_ / (
            "hidden_states_prenorm" if args.hidden_states else "generation_cache"
        )
        problem_file = run_output_dir_ / "problems.jsonl"

        if scored_predictions_path.exists() and not args.overwrite_samples:
            print(f"Found existing scored predictions at {scored_predictions_path}, skipping.")
            continue

        os.makedirs(run_output_dir_, exist_ok=True)
        write_problem_file(base_test_df.to_dict(orient="records"), problem_file)

        predictions = None
        if not args.overwrite_samples:
            predictions = load_complete_predictions(predictions_path, expected_rows)
        if predictions is not None:
            print(f"Found existing predictions at {predictions_path}, skipping generation.")
        else:
            predictions = evaluate_gsm8k_python(
                model=model,
                tokenizer=tokenizer,
                test_df=test_df,
                batch_size=args.eval_batch_size,
                typo_p=typo_p,
                use_sampling=(typo_p == 0.0 and args.sample),
                generation_cache_dir=generation_cache_dir,
                return_hidden_states=args.hidden_states,
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
        for key, value in metrics.items():
            print(f"{key}: {value}")

        results = pd.read_json(scored_predictions_path, lines=True)
        with open(run_output_dir_ / "example_prompt.txt", "w") as fo:
            fo.write(results.iloc[0]["prompt"])
        with open(run_output_dir_ / "metrics.json", "w") as fo:
            json.dump(metrics, fo, indent=4)


if __name__ == "__main__":
    main()
