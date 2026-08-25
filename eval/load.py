from __future__ import annotations

from functools import lru_cache
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from transformers import AutoTokenizer

from eval.paths import MODALITY_RESULTS_DIRS, get_results_roots


VariantType = Literal["retok", "typo", "temperature"]
EXPECTED_NONZERO_P_VALUES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_RESULTS_ROOTS: Tuple[Path, ...] = get_results_roots()
MODALITY_DIR_NAMES = MODALITY_RESULTS_DIRS
RAW_RESULT_RE = re.compile(
    r"^(?P<dataset>.+)_N_(?P<dataset_size>\d+)_(?P<count_label>numretokenizations|numvariants)_(?P<numvariants>\d+)_"
    r"(?:(?P<label>typop|retokp)_)?(?P<p>\d+(?:\.\d+)?)(?P<temp_ident>.*?)(?P<sampled>_sampled)?\.df$"
)
STRUCTURED_RUN_RE = re.compile(
    r"^(?P<label>retokp|typop)_(?P<p>\d+(?:\.\d+)?)_maxexamples_(?P<dataset_size>\d+)_unbiasedsize_(?P<numvariants>\d+)"
    r"(?P<temp_ident>.*)$"
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    dataset: str
    model_name: str
    modality: Literal["retok", "typo"]
    dataset_size: int
    numvariants: int
    p_value: float
    temp_ident: str
    sampled_p0: bool
    storage_kind: Literal["raw_hdf", "structured_jsonl"]

    @property
    def actual_variants_per_task(self) -> int:
        if self.p_value == 0.0 and self.sampled_p0:
            return self.numvariants * 5
        if self.p_value == 0.0:
            return 1
        return self.numvariants


def _canonical_model_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def _canonical_p_value(value: float) -> float:
    return round(float(value), 2)


def _format_temperature_suffix(temperature: float) -> str:
    if float(temperature) == 1.0:
        return ""

    temperature_str = f"{float(temperature):.6f}".rstrip("0").rstrip(".")
    temperature_str = temperature_str.replace("-", "neg").replace(".", "p")
    return f"_temp_{temperature_str}"


def _append_temperature_suffix(temp_ident: str, temperature: float) -> str:
    suffix = _format_temperature_suffix(temperature)
    if not suffix or temp_ident.endswith(suffix):
        return temp_ident
    return f"{temp_ident}{suffix}"


@lru_cache(maxsize=None)
def _mmlu_answer_token_id_map(model_name: str) -> dict[str, set[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    token_id_map: dict[str, set[int]] = {}
    for answer in "ABCD":
        token_ids = set()
        convert_id = tokenizer.convert_tokens_to_ids(answer)
        if isinstance(convert_id, int) and convert_id >= 0:
            token_ids.add(convert_id)
        encoded = tokenizer.encode(answer, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
        token_id_map[answer] = token_ids
    return token_id_map


@lru_cache(maxsize=None)
def _load_tokenizer_local(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _decode_token_ids(model_name: str, token_ids) -> str:
    tokenizer = _load_tokenizer_local(model_name)
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _extract_mmlu_option_labels_from_prompt(prompt_text: str) -> list[str]:
    labels = re.findall(r"(?m)^([A-Za-z]+)\.\s", prompt_text)
    return [label.upper() for label in labels[:4]]


def _extract_generated_label_key_from_text(generated_text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z]+", generated_text)
    if match is None:
        return None
    return match.group(0)[0].upper()


def _maybe_cache_legacy_correctness_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "passed" in df.columns and "cached_passed" not in df.columns:
        df["cached_passed"] = df["passed"]
    if "Correct" in df.columns and "cached_Correct" not in df.columns:
        df["cached_Correct"] = df["Correct"]
    return df


def _mmlu_label_key_from_token_id(model_name: str, token_id) -> Optional[str]:
    try:
        answer_token_id_map = _mmlu_answer_token_id_map(model_name)
    except Exception:
        return None

    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None

    decoded = _decode_token_ids(model_name, [token_id])
    key = _extract_generated_label_key_from_text(decoded)
    if key is not None:
        return key

    for answer_letter in "ABCD":
        if token_id in answer_token_id_map[answer_letter]:
            return answer_letter
    return None


def _series_of_none(index) -> pd.Series:
    return pd.Series([None] * len(index), index=index, dtype=object)


def _first_token_label_keys(model_name: str, values: pd.Series) -> pd.Series:
    def _first_token_key(token_ids) -> Optional[str]:
        if not isinstance(token_ids, (list, tuple)) or len(token_ids) == 0:
            return None
        return _mmlu_label_key_from_token_id(model_name, token_ids[0])

    return values.map(_first_token_key)


def _rescore_mmlu_typo_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = _maybe_cache_legacy_correctness_columns(df)
    if "prompt_tokens" not in df.columns or "answer" not in df.columns:
        return df

    prompt_texts = df["prompt_tokens"].map(lambda token_ids: _decode_token_ids(model_name, token_ids))
    option_labels = prompt_texts.map(_extract_mmlu_option_labels_from_prompt)
    canonical_indices = df["answer"].astype(str).str.strip().map(lambda answer: "ABCD".find(answer))
    ambiguous_option_labels = option_labels.map(
        lambda labels: len(labels) != 4 or len({label[0] for label in labels if label}) != 4
    )
    generated_label_keys = _series_of_none(df.index)
    if "generated_tokens" in df.columns:
        generated_label_keys = _first_token_label_keys(model_name, df["generated_tokens"])

    if "top25_indices" in df.columns:
        missing_mask = generated_label_keys.isna()
        if missing_mask.any():
            generated_label_keys.loc[missing_mask] = _first_token_label_keys(
                model_name,
                df.loc[missing_mask, "top25_indices"],
            )

    correct_label_keys = []
    for labels, canonical_index in zip(option_labels.tolist(), canonical_indices.tolist()):
        if canonical_index is None or canonical_index < 0 or canonical_index >= len(labels):
            correct_label_keys.append(None)
            continue
        label = labels[canonical_index]
        correct_label_keys.append(label[0].upper() if label else None)

    correct_label_keys = pd.Series(correct_label_keys, index=df.index, dtype=object)
    rescored_passed = (
        (~ambiguous_option_labels)
        & generated_label_keys.notna()
        & correct_label_keys.notna()
        & (generated_label_keys == correct_label_keys)
    ).astype(int)

    df["typo_option_labels"] = option_labels
    df["typo_generated_label_key"] = generated_label_keys
    df["typo_correct_label_key"] = correct_label_keys
    df["typo_option_label_ambiguous"] = ambiguous_option_labels.astype(bool)
    df["passed"] = rescored_passed
    return df


def _rescore_mmlu_non_typo_df(df: pd.DataFrame, model_name: str, artifact_path: Path) -> pd.DataFrame:
    if "answer" not in df.columns:
        return df

    predicted_answers = _series_of_none(df.index)
    if "generated_tokens" in df.columns:
        predicted_answers = _first_token_label_keys(model_name, df["generated_tokens"])

    if "top25_indices" in df.columns:
        missing_mask = predicted_answers.isna()
        if missing_mask.any():
            predicted_answers.loc[missing_mask] = _first_token_label_keys(
                model_name,
                df.loc[missing_mask, "top25_indices"],
            )

    if "ABCD_probs" in df.columns:
        answer_letters = list("ABCD")

        def _answer_from_abcd_probs(probs) -> Optional[str]:
            if not isinstance(probs, (list, tuple)) or len(probs) != 4:
                raise ValueError(
                    f"Could not derive 'passed' from ABCD_probs in artifact {artifact_path}: "
                    f"expected a length-4 list/tuple, got {type(probs)} with value {probs!r}."
                )
            return answer_letters[int(max(range(4), key=lambda idx: probs[idx]))]

        missing_mask = predicted_answers.isna()
        if missing_mask.any():
            predicted_answers.loc[missing_mask] = df.loc[missing_mask, "ABCD_probs"].map(_answer_from_abcd_probs)

    if predicted_answers.notna().all():
        df["passed"] = (predicted_answers == df["answer"].astype(str).str.strip()).astype(int)
    return df


def _normalize_correctness_columns(
    df: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    source_artifact: Artifact,
) -> pd.DataFrame:
    if dataset == "mmlu" and source_artifact.modality == "typo":
        return _rescore_mmlu_typo_df(df, model_name=model_name)

    if dataset == "mmlu":
        df = _rescore_mmlu_non_typo_df(df, model_name=model_name, artifact_path=source_artifact.path)

    if "Correct" in df.columns and "passed" not in df.columns:
        df = df.rename(columns={"Correct": "passed"})
    if "passed" not in df.columns and "generated_answer" in df.columns and "answer" in df.columns:
        df["passed"] = (df["generated_answer"] == df["answer"]).astype(int)
    return df


def _expected_raw_path(
    *,
    root: Path,
    dataset: str,
    model_name: str,
    modality: Literal["retok", "typo"],
    dataset_size: int,
    numvariants: int,
    p_value: float,
    temp_ident: str,
    sampled_p0: bool,
) -> Path:
    results_dir_name = MODALITY_DIR_NAMES[modality]
    model_dir = root / results_dir_name / dataset / _canonical_model_dir(model_name)
    count_label = "numretokenizations" if modality == "retok" else "numvariants"
    base_name = f"{dataset}_N_{dataset_size}_{count_label}_{numvariants}_"
    if modality == "typo":
        base_name += "typop_"
    file_name = f"{base_name}{p_value:.2f}{temp_ident}.df"
    if p_value == 0.0 and sampled_p0:
        file_name = file_name[:-3] + "_sampled.df"
    return model_dir / file_name


def _expected_structured_path(
    *,
    root: Path,
    dataset: str,
    model_name: str,
    modality: Literal["retok", "typo"],
    dataset_size: int,
    numvariants: int,
    p_value: float,
    temp_ident: str,
    sampled_p0: bool,
) -> Path:
    results_dir_name = MODALITY_DIR_NAMES[modality]
    model_dir = root / results_dir_name / dataset / _canonical_model_dir(model_name)
    label = "retokp" if modality == "retok" else "typop"
    run_dir = model_dir / f"{label}_{p_value}_maxexamples_{dataset_size}_unbiasedsize_{numvariants}{temp_ident}"
    if p_value == 0.0 and not sampled_p0:
        run_dir = run_dir / "dontsample"
    return run_dir / "scored_predictions.jsonl"


def _format_missing_paths_error(header: str, paths: Sequence[Path]) -> str:
    unique_paths = []
    seen = set()
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique_paths.append(path_str)
    formatted = "\n".join(f"  - {path}" for path in unique_paths)
    return f"{header}\nExpected one of these paths:\n{formatted}"


def _missing_temperature_paths(
    *,
    roots: Sequence[Path],
    dataset: str,
    model_name: str,
    dataset_size: int,
    numvariants: int,
    temp_ident: str,
    source_preference: Sequence[str],
) -> list[Path]:
    paths = []
    for root in roots:
        for modality in source_preference:
            if modality not in MODALITY_DIR_NAMES:
                continue
            paths.append(
                _expected_raw_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=0.0,
                    temp_ident=temp_ident,
                    sampled_p0=True,
                )
            )
            paths.append(
                _expected_structured_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=0.0,
                    temp_ident=temp_ident,
                    sampled_p0=True,
                )
            )
    return paths


def _missing_nonzero_paths(
    *,
    roots: Sequence[Path],
    dataset: str,
    model_name: str,
    modality: Literal["retok", "typo"],
    dataset_size: int,
    numvariants: int,
    temp_ident: str,
) -> list[Path]:
    paths = []
    for root in roots:
        for p_value in EXPECTED_NONZERO_P_VALUES:
            paths.append(
                _expected_raw_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=p_value,
                    temp_ident=temp_ident,
                    sampled_p0=False,
                )
            )
            paths.append(
                _expected_structured_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=p_value,
                    temp_ident=temp_ident,
                    sampled_p0=False,
                )
            )
    return paths


def _missing_greedy_p0_paths(
    *,
    roots: Sequence[Path],
    dataset: str,
    model_name: str,
    preferred_modality: Literal["retok", "typo"],
    dataset_size: int,
    numvariants: int,
    temp_ident: str,
) -> list[Path]:
    paths = []
    modalities = (preferred_modality, "typo" if preferred_modality == "retok" else "retok")
    for root in roots:
        for modality in modalities:
            paths.append(
                _expected_raw_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=0.0,
                    temp_ident=temp_ident,
                    sampled_p0=False,
                )
            )
            paths.append(
                _expected_structured_path(
                    root=root,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=0.0,
                    temp_ident=temp_ident,
                    sampled_p0=False,
                )
            )
    return paths


def _iter_raw_artifacts(
    *,
    root: Path,
    dataset: str,
    model_name: str,
) -> Iterable[Artifact]:
    model_dir_name = _canonical_model_dir(model_name)
    for modality, results_dir_name in MODALITY_DIR_NAMES.items():
        model_dir = root / results_dir_name / dataset / model_dir_name
        if not model_dir.exists():
            continue
        for path in sorted(model_dir.glob("*.df")):
            match = RAW_RESULT_RE.match(path.name)
            if not match:
                continue
            if match.group("dataset") != dataset:
                continue
            count_label = match.group("count_label")
            if modality == "retok" and count_label != "numretokenizations":
                continue
            if modality == "typo" and count_label != "numvariants":
                continue
            label = match.group("label")
            if label == "retokp" and modality != "retok":
                continue
            if label == "typop" and modality != "typo":
                continue
            yield Artifact(
                path=path,
                dataset=dataset,
                model_name=model_name,
                modality=modality,
                dataset_size=int(match.group("dataset_size")),
                numvariants=int(match.group("numvariants")),
                p_value=_canonical_p_value(match.group("p")),
                temp_ident=match.group("temp_ident") or "",
                sampled_p0=bool(match.group("sampled")),
                storage_kind="raw_hdf",
            )


def _iter_structured_artifacts(
    *,
    root: Path,
    dataset: str,
    model_name: str,
) -> Iterable[Artifact]:
    model_dir_name = _canonical_model_dir(model_name)
    for modality, results_dir_name in MODALITY_DIR_NAMES.items():
        model_dir = root / results_dir_name / dataset / model_dir_name
        if not model_dir.exists():
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            match = STRUCTURED_RUN_RE.match(run_dir.name)
            if not match:
                continue
            label = match.group("label")
            if label == "retokp" and modality != "retok":
                continue
            if label == "typop" and modality != "typo":
                continue

            p_value = _canonical_p_value(match.group("p"))
            dataset_size = int(match.group("dataset_size"))
            numvariants = int(match.group("numvariants"))
            temp_ident = match.group("temp_ident") or ""

            sampled_path = run_dir / "scored_predictions.jsonl"
            if sampled_path.exists():
                yield Artifact(
                    path=sampled_path,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=p_value,
                    temp_ident=temp_ident,
                    sampled_p0=(p_value == 0.0),
                    storage_kind="structured_jsonl",
                )

            dontsample_path = run_dir / "dontsample" / "scored_predictions.jsonl"
            if p_value == 0.0 and dontsample_path.exists():
                yield Artifact(
                    path=dontsample_path,
                    dataset=dataset,
                    model_name=model_name,
                    modality=modality,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    p_value=p_value,
                    temp_ident=temp_ident,
                    sampled_p0=False,
                    storage_kind="structured_jsonl",
                )


def _discover_artifacts(
    *,
    dataset: str,
    model_name: str,
    roots: Optional[Sequence[Path]] = None,
) -> list[Artifact]:
    roots = tuple(Path(root) for root in (roots or DEFAULT_RESULTS_ROOTS))
    artifacts: list[Artifact] = []
    for root in roots:
        artifacts.extend(_iter_raw_artifacts(root=root, dataset=dataset, model_name=model_name))
        artifacts.extend(_iter_structured_artifacts(root=root, dataset=dataset, model_name=model_name))
    return artifacts


def _keep_first_artifact_per_key(artifacts: Sequence[Artifact]) -> list[Artifact]:
    deduped: list[Artifact] = []
    seen = set()
    for artifact in artifacts:
        key = (
            artifact.modality,
            artifact.dataset_size,
            artifact.numvariants,
            artifact.p_value,
            artifact.temp_ident,
            artifact.sampled_p0,
            artifact.storage_kind,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(artifact)
    return deduped


def _select_greedy_p0_artifact(
    candidates: Sequence[Artifact],
    *,
    preferred_modality: Literal["retok", "typo"],
    requested_numvariants: int,
) -> Artifact:
    if not candidates:
        raise FileNotFoundError("No greedy p=0 artifact found.")

    def score(artifact: Artifact) -> tuple[int, int]:
        modality_penalty = 0 if artifact.modality == preferred_modality else 1
        count_penalty = 0 if artifact.numvariants == requested_numvariants else 1
        return (modality_penalty, count_penalty)

    return min(candidates, key=score)


def _select_temperature_artifact(
    candidates: Sequence[Artifact],
    *,
    requested_numvariants: int,
    source_preference: Sequence[str],
) -> Artifact:
    if not candidates:
        raise FileNotFoundError("No sampled p=0 artifact found for temperature/pass@k loading.")

    preference_rank = {name: idx for idx, name in enumerate(source_preference)}

    def score(artifact: Artifact) -> tuple[int, int, int]:
        return (
            abs(artifact.actual_variants_per_task - requested_numvariants),
            preference_rank.get(artifact.modality, len(preference_rank)),
            abs(artifact.numvariants - requested_numvariants),
        )

    return min(candidates, key=score)


def _read_artifact(artifact: Artifact) -> pd.DataFrame:
    if artifact.storage_kind == "raw_hdf":
        return pd.read_hdf(artifact.path, key="df")
    return pd.read_json(artifact.path, lines=True)


def _normalize_loaded_df(
    df: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    requested_variant_type: VariantType,
    source_artifact: Artifact,
) -> pd.DataFrame:
    df = df.copy()
    df = _normalize_correctness_columns(
        df,
        dataset=dataset,
        model_name=model_name,
        source_artifact=source_artifact,
    )
    if "passed" not in df.columns:
        raise ValueError(f"Loaded artifact at {source_artifact.path} does not contain a 'passed' column.")

    if "task_id" not in df.columns:
        if "prompti" in df.columns:
            df["task_id"] = df["prompti"].map(lambda idx: f"{dataset}/{idx}")
        else:
            raise ValueError(f"Loaded artifact at {source_artifact.path} has neither 'task_id' nor 'prompti'.")

    if "p" not in df.columns:
        df["p"] = source_artifact.p_value

    df["passed"] = df["passed"].astype(int)
    df["source_p"] = source_artifact.p_value
    df["dataset"] = dataset
    df["model_name"] = model_name
    df["requested_variant_type"] = requested_variant_type
    df["source_variant_type"] = source_artifact.modality
    df["sampled_p0"] = source_artifact.sampled_p0
    df["source_path"] = str(source_artifact.path)
    df["numvariants_config"] = source_artifact.numvariants
    df["is_canonical_variant"] = bool(source_artifact.p_value == 0.0 and not source_artifact.sampled_p0)
    df["injected_canonical"] = False

    return df


def _load_artifacts(
    artifacts: Sequence[Artifact],
    *,
    dataset: str,
    model_name: str,
    requested_variant_type: VariantType,
) -> pd.DataFrame:
    if not artifacts:
        raise FileNotFoundError("No matching artifacts to load.")

    frames = [
        _normalize_loaded_df(
            _read_artifact(artifact),
            dataset=dataset,
            model_name=model_name,
            requested_variant_type=requested_variant_type,
            source_artifact=artifact,
        )
        for artifact in artifacts
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["p", "task_id"], kind="stable").reset_index(drop=True)
    return df


def _attach_metadata(
    df: pd.DataFrame,
    *,
    dataset: str,
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    artifacts: Sequence[Artifact],
) -> pd.DataFrame:
    df.attrs["dataset"] = dataset
    df.attrs["model_name"] = model_name
    df.attrs["dataset_size"] = dataset_size
    df.attrs["requested_numvariants"] = numvariants
    df.attrs["variant_type"] = variant_type
    df.attrs["loaded_p_values"] = tuple(sorted(_canonical_p_value(p) for p in df["p"].unique()))
    df.attrs["source_paths"] = tuple(str(artifact.path) for artifact in artifacts)
    df.attrs["source_modalities"] = tuple(dict.fromkeys(artifact.modality for artifact in artifacts))
    counts = df.groupby("task_id").size()
    df.attrs["variants_per_task_min"] = int(counts.min())
    df.attrs["variants_per_task_max"] = int(counts.max())
    return df


def load_dataset(
    *,
    dataset: str,
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    temp_ident: str = "",
    roots: Optional[Sequence[Union[str, Path]]] = None,
    temperature_source_preference: Sequence[str] = ("retok", "typo"),
) -> pd.DataFrame:
    """
    Load one consolidated dataframe for a dataset/model/variant family.

    `variant_type="retok"` and `variant_type="typo"` load all matching nonzero-p
    runs for that family plus one greedy, unsampled p=0 slice.
    `variant_type="temperature"` loads one sampled p=0 artifact, chosen to be as
    close as possible to the requested number of variants.
    """
    if variant_type not in {"retok", "typo", "temperature"}:
        raise ValueError(f"Unknown variant_type: {variant_type}")

    resolved_roots = tuple(Path(root) for root in (roots or DEFAULT_RESULTS_ROOTS))
    artifacts = _keep_first_artifact_per_key(
        [
            artifact
            for artifact in _discover_artifacts(
                dataset=dataset,
                model_name=model_name,
                roots=resolved_roots,
            )
            if artifact.dataset_size == dataset_size and artifact.temp_ident == temp_ident
        ]
    )

    if variant_type == "temperature":
        if dataset == "mmlu":
            greedy_candidates = [
                artifact
                for artifact in artifacts
                if artifact.p_value == 0.0 and not artifact.sampled_p0
            ]
            if not greedy_candidates:
                raise FileNotFoundError(
                    _format_missing_paths_error(
                        (
                            f"No greedy p=0 MMLU artifact with answer probabilities found for dataset={dataset}, "
                            f"model={model_name}, dataset_size={dataset_size}, requested_numvariants={numvariants}, "
                            f"temp_ident={temp_ident!r}."
                        ),
                        _missing_greedy_p0_paths(
                            roots=resolved_roots,
                            dataset=dataset,
                            model_name=model_name,
                            preferred_modality="retok",
                            dataset_size=dataset_size,
                            numvariants=numvariants,
                            temp_ident=temp_ident,
                        ),
                    )
                )
            chosen = _select_greedy_p0_artifact(
                greedy_candidates,
                preferred_modality="retok",
                requested_numvariants=numvariants,
            )
        else:
            temp_candidates = [artifact for artifact in artifacts if artifact.p_value == 0.0 and artifact.sampled_p0]
            if not temp_candidates:
                raise FileNotFoundError(
                    _format_missing_paths_error(
                        (
                            f"No sampled p=0 artifact found for dataset={dataset}, model={model_name}, "
                            f"dataset_size={dataset_size}, requested_numvariants={numvariants}, temp_ident={temp_ident!r}."
                        ),
                        _missing_temperature_paths(
                            roots=resolved_roots,
                            dataset=dataset,
                            model_name=model_name,
                            dataset_size=dataset_size,
                            numvariants=numvariants,
                            temp_ident=temp_ident,
                            source_preference=temperature_source_preference,
                        ),
                    )
                )
            chosen = _select_temperature_artifact(
                temp_candidates,
                requested_numvariants=numvariants,
                source_preference=temperature_source_preference,
            )
        df = _load_artifacts([chosen], dataset=dataset, model_name=model_name, requested_variant_type=variant_type)
        return _attach_metadata(
            df,
            dataset=dataset,
            model_name=model_name,
            dataset_size=dataset_size,
            numvariants=numvariants,
            variant_type=variant_type,
            artifacts=[chosen],
        )

    nonzero_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.modality == variant_type
        and artifact.numvariants == numvariants
        and artifact.p_value > 0.0
        and not artifact.sampled_p0
    ]
    if not nonzero_artifacts:
        raise FileNotFoundError(
            _format_missing_paths_error(
                (
                    f"No nonzero-{variant_type} artifacts found for dataset={dataset}, model={model_name}, "
                    f"dataset_size={dataset_size}, numvariants={numvariants}, temp_ident={temp_ident!r}."
                ),
                _missing_nonzero_paths(
                    roots=resolved_roots,
                    dataset=dataset,
                    model_name=model_name,
                    modality=variant_type,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    temp_ident=temp_ident,
                ),
            )
        )

    p0_candidates = [
        artifact
        for artifact in artifacts
        if artifact.p_value == 0.0 and not artifact.sampled_p0
    ]
    if not p0_candidates:
        raise FileNotFoundError(
            _format_missing_paths_error(
                (
                    f"No greedy p=0 donor artifact found for dataset={dataset}, model={model_name}, "
                    f"dataset_size={dataset_size}, preferred_variant_type={variant_type}, "
                    f"requested_numvariants={numvariants}, temp_ident={temp_ident!r}."
                ),
                _missing_greedy_p0_paths(
                    roots=resolved_roots,
                    dataset=dataset,
                    model_name=model_name,
                    preferred_modality=variant_type,
                    dataset_size=dataset_size,
                    numvariants=numvariants,
                    temp_ident=temp_ident,
                ),
            )
        )
    p0_artifact = _select_greedy_p0_artifact(
        p0_candidates,
        preferred_modality=variant_type,
        requested_numvariants=numvariants,
    )

    selected_artifacts = sorted(
        [p0_artifact, *nonzero_artifacts],
        key=lambda artifact: (artifact.p_value, artifact.path.name),
    )
    df = _load_artifacts(
        selected_artifacts,
        dataset=dataset,
        model_name=model_name,
        requested_variant_type=variant_type,
    )
    return _attach_metadata(
        df,
        dataset=dataset,
        model_name=model_name,
        dataset_size=dataset_size,
        numvariants=numvariants,
        variant_type=variant_type,
        artifacts=selected_artifacts,
    )


def load_gsm8k(
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper around `load_dataset(..., dataset=\"gsm8k\")`."""
    return load_dataset(
        dataset="gsm8k",
        model_name=model_name,
        dataset_size=dataset_size,
        numvariants=numvariants,
        variant_type=variant_type,
        **kwargs,
    )


def load_mmlu(
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper around `load_dataset(..., dataset=\"mmlu\")`."""
    return load_dataset(
        dataset="mmlu",
        model_name=model_name,
        dataset_size=dataset_size,
        numvariants=numvariants,
        variant_type=variant_type,
        **kwargs,
    )


def load_humaneval(
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    temperature: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper around `load_dataset(..., dataset=\"humaneval\")`."""
    if variant_type == "temperature":
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        kwargs["temp_ident"] = _append_temperature_suffix(kwargs.get("temp_ident", ""), temperature)

    return load_dataset(
        dataset="humaneval",
        model_name=model_name,
        dataset_size=dataset_size,
        numvariants=numvariants,
        variant_type=variant_type,
        **kwargs,
    )


def load_gsm8k_python(
    model_name: str,
    dataset_size: int,
    numvariants: int,
    variant_type: VariantType,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper around `load_dataset(..., dataset="gsm8k_python")`."""
    return load_dataset(
        dataset="gsm8k_python",
        model_name=model_name,
        dataset_size=dataset_size,
        numvariants=numvariants,
        variant_type=variant_type,
        **kwargs,
    )
