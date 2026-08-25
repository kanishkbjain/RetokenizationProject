import json
import random
import textwrap
from pathlib import Path

from eval.humaneval.data import write_jsonl


DATA_PATH = Path(__file__).resolve().parent.parent / "gsm8k" / "full_gsm8k_test.jsonl"
DEFAULT_ABS_TOLERANCE = 1e-6


def parse_ground_truth_answer(answer_text):
    final_answer = answer_text.split("####")[-1].strip().replace(",", "")
    return float(final_answer)


def _escape_docstring(text):
    return text.replace('"""', '\\"\\"\\"')


def format_python_prompt(question, width=60):
    escaped_question = _escape_docstring(question.strip())
    wrapped_lines = textwrap.wrap(
        escaped_question,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped_lines:
        wrapped_lines = [""]

    if len(wrapped_lines) == 1:
        docstring_lines = [f'    """{wrapped_lines[0]}"""']
    else:
        docstring_lines = [f'    """{wrapped_lines[0]}']
        docstring_lines.extend(f"    {line}" for line in wrapped_lines[1:-1])
        docstring_lines.append(f'    {wrapped_lines[-1]}"""')

    return "def function():\n" + "\n".join(docstring_lines) + "\n"


def build_numeric_test(answer, abs_tolerance=DEFAULT_ABS_TOLERANCE):
    return f"""import math


def check(candidate):
    result = candidate()
    assert not isinstance(result, bool), f"Expected numeric return, got bool: {{result!r}}"
    assert isinstance(result, (int, float)), (
        f"Expected numeric return, got {{type(result).__name__}}: {{result!r}}"
    )
    expected = {answer!r}
    numeric_result = float(result)
    assert math.isfinite(numeric_result), f"Expected finite numeric return, got {{result!r}}"
    assert math.isclose(
        numeric_result,
        expected,
        rel_tol=0.0,
        abs_tol={abs_tolerance!r},
    ), f"Expected {{expected}}, got {{result!r}}"
"""


def load_gsm8k_examples(max_num_examples=None):
    examples = []
    with DATA_PATH.open("r", encoding="utf-8") as fi:
        for idx, line in enumerate(fi):
            record = json.loads(line)
            record["_source_index"] = idx
            examples.append(record)

    if max_num_examples is not None:
        sample_size = min(len(examples), max_num_examples)
        examples = random.Random(42).sample(examples, sample_size)

    return examples


def build_problem_records(examples, abs_tolerance=DEFAULT_ABS_TOLERANCE):
    problem_records = []
    for idx, example in enumerate(examples):
        answer = parse_ground_truth_answer(example["answer"])
        source_index = example.get("_source_index", example.get("source_index", idx))
        task_id = f"GSM8K-Python/{source_index}"
        problem_records.append(
            {
                "task_id": task_id,
                "prompt": format_python_prompt(example["question"]),
                "entry_point": "function",
                "test": build_numeric_test(answer, abs_tolerance=abs_tolerance),
                "question": example["question"],
                "expected_answer": answer,
                "source_index": source_index,
            }
        )
    return problem_records


def _is_body_line(line, allow_two_space_indentation=False):
    return (
        line.startswith("    ")
        or line.startswith("\t")
        or (allow_two_space_indentation and line.startswith("  "))
    )


def _count_docstring_delimiters(line):
    return line.count('"""') + line.count("'''")


def extract_code_from_generation(generation, allow_two_space_indentation=False):
    raw_lines = generation.splitlines()
    lines = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            fence_suffix = stripped[3:].strip()
            if fence_suffix.lower() in {"", "python"}:
                continue
            line = fence_suffix
        if line.rstrip().endswith("```"):
            line = line.rstrip()[:-3].rstrip()
        lines.append(line)

    extracted_lines = []
    start_idx = 0
    saw_function_signature = False

    for idx, line in enumerate(lines):
        if line.startswith("def function():"):
            saw_function_signature = True
            extracted_lines.append(line)
            triple_quote_count = 0
            for body_start_idx in range(idx + 1, len(lines)):
                extracted_lines.append(lines[body_start_idx])
                triple_quote_count += _count_docstring_delimiters(lines[body_start_idx])
                if triple_quote_count >= 2:
                    start_idx = body_start_idx + 1
                    break
            break

    body_started = False

    for line in lines[start_idx:]:
        stripped = line.strip()

        if stripped.startswith("```"):
            if body_started:
                break
            continue

        if not body_started and stripped == "":
            continue

        if _is_body_line(line, allow_two_space_indentation=allow_two_space_indentation):
            extracted_lines.append(line)
            body_started = True
            continue

        if body_started:
            if stripped == "":
                extracted_lines.append(line)
                continue
            break

    if not saw_function_signature:
        extracted_lines = [line for line in extracted_lines if line.strip() != ""]

    extracted = "\n".join(extracted_lines).rstrip()
    return extracted + "\n"


def write_problem_file(problem_records, output_path):
    serializable_records = [
        {
            "task_id": record["task_id"],
            "prompt": record["prompt"],
            "entry_point": record["entry_point"],
            "test": record["test"],
        }
        for record in problem_records
    ]
    write_jsonl(str(output_path), serializable_records)
