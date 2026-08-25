import re
import os
import json
from pathlib import Path

import click
import pandas as pd

from eval.humaneval.evaluation import evaluate_functional_correctness
from eval.util import batched_generate_legacy, load_model_and_tokenizer
from olmo.torch_util import seed_all

seed_all(42)

HUMANEVAL_DIR = Path(__file__).resolve().parent
HUMANEVAL_JSON_PATH = HUMANEVAL_DIR / "HumanEval.jsonl"


def get_output(out):
    lines = out.splitlines()
    lines_ = []
    comment_done = False
    for line in lines:
        lines_.append(line)
        if comment_done:
            if '    return' == line[:10]:
                break
        else:
            if line == '    """':
                comment_done = True
    return '\n'.join(lines_)

def evaluate_humaneval(
    model,
    tokenizer,
    test_df,
    batch_size,
    retokenizationp,
        dont_sample=False,
    return_hidden_states=False,
    hidden_states_dir=None,
):
    test_df = test_df.reset_index(drop=True)
    prompts = test_df.prompt.tolist()

    print(f"--- HumanEval example prompt ---\n{prompts[0]}\n----------------------")

    generation_kwargs = {'do_sample': False, 'max_new_tokens': 500}

    if retokenizationp == 0.0 and not dont_sample:
        generation_kwargs = {'do_sample': True, 'top_p': 0.9, 'temperature': 1.0, 'max_new_tokens': 500}
        print('Using sampling generation kwargs for retokenizationp=0.0', generation_kwargs)

    outputs_generated = batched_generate_legacy(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        retokenizationp=retokenizationp,
        return_last_hidden_states=return_hidden_states,
        hidden_states_dir=hidden_states_dir,
        store_hidden_states_in_memory=False,
        hidden_states_pre_norm=True,
        **generation_kwargs,
    )
    # remove stop_strings if they are at the end of the string
    outputs = [get_output(out) for out in outputs_generated['generations']]

    if return_hidden_states:
        results = [{
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
    else:
        results = [
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
    return results


@click.command()
@click.option("--model_name_or_path", type=str, default=None)
@click.option("--output_dir", type=str)
@click.option("--step", type=int, default=None)
@click.option("--max_num_examples", type=int, default=None)
@click.option("--eval_batch_size", type=int, default=64)
@click.option("--pass_at_k", type=int, default=10)
@click.option("--unbiased_sampling_size_n", type=int, default=20)
@click.option("--overwrite_samples", is_flag=False, default=False)
@click.option("--pvals", type=str, default='0.0,0.2,0.4,0.6,0.8,1.0')
@click.option("--dont_sample", is_flag=False, default=False)
@click.option("--hidden_states", is_flag=False, default=False)
def main(
    model_name_or_path: str,
    output_dir: str,
    step: int,
    max_num_examples: int,
    eval_batch_size: int,
    pass_at_k: int,
    unbiased_sampling_size_n: int,
    overwrite_samples: bool,
    pvals: str,
    dont_sample: bool,
    hidden_states: bool = False,
):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    output_dir = Path(output_dir)
    # add model name to output dir if not present
    model_dir_name = model_name_or_path.replace("/", "_")
    output_dir = output_dir / model_dir_name
    print(output_dir)
    retokenizationps = [float(x) for x in pvals.split(",") if x.strip() != ""]

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, step=step)
    test_df = pd.read_json(HUMANEVAL_JSON_PATH, lines=True)


    if max_num_examples:
        if not os.path.exists(f'eval/humaneval/sampled_data_{max_num_examples}.h5'):
            test_df = test_df.sample(min(len(test_df), max_num_examples), random_state=42)
            test_df.to_hdf(f'eval/humaneval/sampled_data_{max_num_examples}.h5', key='df', mode='w')
            print('Sampled and saved new test data.')
        else:
            print('Loading existing sampled test data.')
            test_df = pd.read_hdf(f'eval/humaneval/sampled_data_{max_num_examples}.h5', key='df')
    else:
        max_num_examples = len(test_df)

    test_df['temp_len'] = test_df['prompt'].apply(len)
    test_df.sort_values('temp_len', inplace=True, ascending=False)
    test_df.drop(columns='temp_len', inplace=True)

    test_df = test_df.reset_index(drop=True)

    if not dont_sample:
        test_df0 = pd.concat([test_df] * unbiased_sampling_size_n*5, ignore_index=True)
    else:
        test_df0 = test_df.copy()

    test_df_all = pd.concat([test_df] * unbiased_sampling_size_n, ignore_index=True)


    for retokenizationp in retokenizationps:
        run_output_dir = output_dir / f"retokp_{retokenizationp}_maxexamples_{max_num_examples}_unbiasedsize_{unbiased_sampling_size_n}"
        if dont_sample:
            run_output_dir = run_output_dir / f"dontsample"
        if not os.path.exists(run_output_dir / "predictions.jsonl") or overwrite_samples:

            # duplicate test_df unbiased_sampling_size_n times
            if retokenizationp == 0.0:
                test_df = test_df0
            else:
                test_df = test_df_all
            hidden_states_dir = run_output_dir / "hidden_states_prenorm"
            predictions = evaluate_humaneval(model, tokenizer, test_df, batch_size=eval_batch_size,
                                             retokenizationp=retokenizationp, dont_sample=dont_sample,
                                             hidden_states_dir=hidden_states_dir, return_hidden_states=hidden_states)

            os.makedirs(run_output_dir, exist_ok=True)
            pd.DataFrame(predictions).to_json(run_output_dir / "predictions.jsonl", orient="records", lines=True)

        if not os.path.exists(run_output_dir / "scored_predictions.jsonl"):
            print(f"Found existing predictions at {run_output_dir / 'predictions.jsonl'}, skipping generation.")
            predictions = pd.read_json(run_output_dir / "predictions.jsonl", lines=True)
            metrics = evaluate_functional_correctness(
                sample_file=str(run_output_dir / "predictions.jsonl"), k=[pass_at_k], n_workers=64
            )
            metrics["num_examples"] = len(predictions) // unbiased_sampling_size_n
            for k, v in metrics.items():
                print(f"{k}: {v}")

            results = pd.read_json(run_output_dir / "scored_predictions.jsonl", lines=True)
            with open(run_output_dir / "example_prompt.txt", "w") as fo:
                fo.write(results.iloc[0]["prompt"])
            with open(run_output_dir / "metrics.json", "w") as fo:
                json.dump(metrics, fo, indent=4)


if __name__ == "__main__":
    main()
