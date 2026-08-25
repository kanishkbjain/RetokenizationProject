from collections import Counter
import random

import numpy as np


TYPO_REPLACEMENTS = {
    "a": ["q", "w", "s", "z", "x", "aa"],
    "b": ["g", "h", "v", "n", "bb"],
    "c": ["d", "f", "x", "v", "cc"],
    "d": ["e", "r", "s", "f", "x", "c", "dd"],
    "e": ["w", "r", "s", "d", "ee"],
    "f": ["r", "t", "d", "g", "c", "v", "ff"],
    "g": ["t", "y", "f", "h", "v", "b", "gg"],
    "h": ["y", "u", "g", "j", "b", "n", "hh"],
    "i": ["u", "o", "j", "k", "ii"],
    "j": ["u", "i", "h", "k", "n", "m", "jj"],
    "k": ["i", "o", "j", "l", "m", "kk"],
    "l": ["o", "p", "k", "ll"],
    "m": ["j", "k", "n", "mm"],
    "n": ["h", "j", "b", "m", "nn"],
    "o": ["i", "p", "k", "l", "oo"],
    "p": ["o", "l", "pp"],
    "q": ["w", "a", "qq"],
    "r": ["e", "t", "d", "f", "rr"],
    "s": ["w", "e", "a", "d", "z", "x", "ss"],
    "t": ["r", "y", "f", "g", "tt"],
    "u": ["y", "i", "h", "j", "uu"],
    "v": ["f", "g", "c", "b", "vv"],
    "w": ["q", "e", "a", "s", "ww"],
    "x": ["s", "d", "z", "c", "xx"],
    "y": ["t", "u", "g", "h", "yy"],
    "z": ["a", "s", "x", "zz"],
}


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


def _decode_token_surface(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _find_docstring_span(text):
    delimiter_candidates = []
    for delimiter in ('"""', "'''"):
        start = text.find(delimiter)
        if start != -1:
            delimiter_candidates.append((start, delimiter))
    if not delimiter_candidates:
        return None

    start, delimiter = min(delimiter_candidates, key=lambda item: item[0])
    content_start = start + len(delimiter)
    end = text.find(delimiter, content_start)
    if end == -1:
        return None
    return content_start, end


def _mutable_alpha_positions(text, token_start=0, allowed_span=None):
    positions = []
    for idx, ch in enumerate(text):
        if ch.lower() not in TYPO_REPLACEMENTS:
            continue
        absolute_idx = token_start + idx
        if allowed_span is not None and not (allowed_span[0] <= absolute_idx < allowed_span[1]):
            continue
        positions.append(idx)
    return positions


def _replace_letter_at_position(word, char_index, have_omission=False):
    if char_index < 0 or char_index >= len(word):
        return word

    char = word[char_index]
    replacement_key = char.lower()
    if replacement_key not in TYPO_REPLACEMENTS:
        return word

    replacement_options = list(TYPO_REPLACEMENTS[replacement_key])
    if have_omission:
        replacement_options.append("")

    random_letter = random.choice(replacement_options)
    if char.isupper():
        random_letter = random_letter.upper()

    return word[:char_index] + random_letter + word[char_index + 1 :]


def _apply_single_typo(surface_form, have_omission=False, candidate_positions=None):
    if candidate_positions is None:
        candidate_positions = _mutable_alpha_positions(surface_form)
    if not candidate_positions:
        return surface_form
    random_position = random.choice(candidate_positions)
    return _replace_letter_at_position(surface_form, random_position, have_omission=have_omission)


def fiddle_tokens_typos(
    inputTensor,
    attention_mask,
    tokenizer,
    is_mcq,
    reason,
    typo_p,
    no_prefix_suffix=False,
    typo_region="all",
):
    inputList = inputTensor.tolist()
    fiddledPrompts = []
    input_ids_prefix, input_ids_suffix = _get_prompt_wrapper_ids(
        tokenizer,
        is_mcq=is_mcq,
        reason=reason,
        no_prefix_suffix=no_prefix_suffix,
    )

    lengths = []
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    for i, prompt in enumerate(inputList):
        real_token_ids = _extract_real_prompt_token_ids(prompt, attention_mask[i], tokenizer)
        token_surfaces = [_decode_token_surface(tokenizer, token_id) for token_id in real_token_ids]
        token_starts = []
        cursor = 0
        for surface in token_surfaces:
            token_starts.append(cursor)
            cursor += len(surface)

        prompt_text = "".join(token_surfaces)
        allowed_span = None
        if typo_region == "docstring":
            allowed_span = _find_docstring_span(prompt_text)
            if allowed_span is None:
                allowed_span = (0, 0)
        elif typo_region != "all":
            raise ValueError(f"Unknown typo_region: {typo_region}")

        eligible_positions = []
        for token_idx, surface in enumerate(token_surfaces):
            if real_token_ids[token_idx] in special_ids:
                continue
            candidate_positions = _mutable_alpha_positions(
                surface,
                token_start=token_starts[token_idx],
                allowed_span=allowed_span,
            )
            if candidate_positions:
                eligible_positions.append(token_idx)

        mutation_counts = None
        if typo_p > 0.0 and eligible_positions:
            num_draws = int(np.random.binomial(len(eligible_positions), typo_p))
            if num_draws > 0:
                sampled_positions = np.random.choice(eligible_positions, size=num_draws, replace=True)
                mutation_counts = Counter(int(pos) for pos in sampled_positions.tolist())
                for token_idx, mutation_count in mutation_counts.items():
                    mutated_surface = token_surfaces[token_idx]
                    for _ in range(mutation_count):
                        candidate_positions = _mutable_alpha_positions(
                            mutated_surface,
                            token_start=token_starts[token_idx],
                            allowed_span=allowed_span,
                        )
                        if not candidate_positions:
                            break
                        mutated_surface = _apply_single_typo(
                            mutated_surface,
                            candidate_positions=candidate_positions,
                        )
                    token_surfaces[token_idx] = mutated_surface

        if not mutation_counts:
            fiddled_prompt_ids = list(real_token_ids)
        else:
            fiddled_prompt_text = "".join(token_surfaces)
            fiddled_prompt_ids = tokenizer.encode(fiddled_prompt_text, add_special_tokens=False)
        lengths.append((len(input_ids_prefix), len(fiddled_prompt_ids), len(input_ids_suffix)))
        fiddledPrompts.append(input_ids_prefix + fiddled_prompt_ids + input_ids_suffix)

    return tokenizer.pad({"input_ids": fiddledPrompts}, padding="longest", padding_side="left", return_tensors="pt"), lengths


def randomly_replace_letter(word, have_omission=False):
    if len(word) == 0:
        return word

    if len(word) == 1 or len(word) == 2:
        return word

    total_length = len(word) + 1

    adding_whitespace = random.random()
    if adding_whitespace <= 1 / total_length:
        return word + " "

    replace_dict = {key: list(values) for key, values in TYPO_REPLACEMENTS.items()}
    if have_omission:
        for key in replace_dict:
            replace_dict[key].append("")

    random_position = random.randint(1, len(word) - 1)

    try:
        if word[random_position].isupper():
            random_letter = replace_dict[word[random_position].lower()][
                random.randint(1, len(replace_dict[word[random_position].lower()]) - 1)
            ]
            random_letter = random_letter.upper()
        else:
            random_letter = replace_dict[word[random_position]][
                random.randint(1, len(replace_dict[word[random_position]]) - 1)
            ]
        return word[:random_position] + random_letter + word[random_position + 1 :]
    except Exception:
        print("except")
        return word
