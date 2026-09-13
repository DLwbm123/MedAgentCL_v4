from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


OPTION_LINE = re.compile(
    r"^\s*([A-Z])\s*[\.\)]\s*(.+?)\s*$",
    flags=re.MULTILINE,
)
LEADING_LETTER = re.compile(
    r"^\s*(?:(?:the\s+)?(?:answer|choice|option)\s*(?:is|:|-)?)?\s*"
    r"([A-Z])(?=[\s\.\):,\-]|$)",
    flags=re.IGNORECASE,
)
NAMED_LETTER = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:|-)?\s*"
    r"([A-Z])(?=[\s\.\):,\-]|$)",
    flags=re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _prompt(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        content = messages[0].get("content", "")
        if isinstance(content, str):
            return content
    for key in ("query", "question", "text"):
        if record.get(key):
            return str(record[key])
    return ""


def extract_options(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    options = record.get("options")
    letters = record.get("option_letters")
    if isinstance(options, list) and options:
        values = [str(item).strip() for item in options]
        if isinstance(letters, list) and len(letters) == len(values):
            keys = [str(item).strip().upper() for item in letters]
        else:
            keys = [chr(ord("A") + index) for index in range(len(values))]
    else:
        matches = OPTION_LINE.findall(_prompt(record))
        keys = [letter.upper() for letter, _ in matches]
        values = [value.strip() for _, value in matches]
    if not values or len(keys) != len(values):
        raise ValueError("Sample has no valid option set")
    if len(set(keys)) != len(keys) or any(len(key) != 1 for key in keys):
        raise ValueError("Sample option letters are invalid")
    return keys, values


def _target(record: Mapping[str, Any], keys: list[str], values: list[str]) -> tuple[str, str]:
    explicit = str(record.get("answer_option") or "").strip().upper()
    if explicit:
        if explicit not in keys:
            raise ValueError("Target answer_option is absent from sample options")
        index = keys.index(explicit)
        return explicit, values[index]
    answer = record.get("answer") or record.get("response")
    if answer is None:
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            answer = messages[-1].get("content")
    normalized = normalize_text(answer)
    matches = [
        index
        for index, value in enumerate(values)
        if normalize_text(value) == normalized
    ]
    if len(matches) != 1:
        raise ValueError("Target text does not map to exactly one sample option")
    index = matches[0]
    return keys[index], values[index]


def _prediction_index(raw: str, keys: list[str], values: list[str]) -> int | None:
    for pattern in (LEADING_LETTER, NAMED_LETTER):
        match = pattern.search(raw)
        if match:
            letter = match.group(1).upper()
            return keys.index(letter) if letter in keys else None

    normalized_raw = normalize_text(raw)
    normalized_options = [normalize_text(value) for value in values]
    exact = [
        index
        for index, value in enumerate(normalized_options)
        if normalized_raw == value
    ]
    if len(exact) == 1:
        return exact[0]
    starts = [
        index
        for index, value in enumerate(normalized_options)
        if normalized_raw.startswith(value + " ")
        or normalized_raw.startswith("the answer is " + value)
        or normalized_raw.startswith("answer " + value)
    ]
    if len(starts) == 1:
        return starts[0]
    contained = [
        index
        for index, value in enumerate(normalized_options)
        if value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized_raw)
    ]
    return contained[0] if len(contained) == 1 else None


def evaluate_prediction(
    record: Mapping[str, Any],
    raw_prediction: str,
) -> dict[str, Any]:
    keys, values = extract_options(record)
    target_letter, target = _target(record, keys, values)
    index = _prediction_index(str(raw_prediction), keys, values)
    valid = index is not None
    normalized_prediction = values[index] if valid else None
    predicted_letter = keys[index] if valid else None
    return {
        "target": target,
        "target_letter": target_letter,
        "raw_prediction": str(raw_prediction),
        "normalized_prediction": normalized_prediction,
        "predicted_letter": predicted_letter,
        "valid": valid,
        "invalid": not valid,
        "correct": bool(valid and predicted_letter == target_letter),
        "option_letters": keys,
        "options": values,
    }


def component_task_ids(
    mode: str,
    *,
    after_task: int,
    eval_task: int,
) -> list[int]:
    if after_task <= 0 or eval_task <= 0 or eval_task > after_task:
        raise ValueError("CL matrix cell must satisfy 1 <= eval_task <= after_task")
    if mode == "primary_cumulative":
        return list(range(1, after_task + 1))
    if mode == "oracle_skill_aware":
        return [eval_task]
    raise ValueError(f"Unknown evaluation mode: {mode}")


def matrix_rows(cells: Sequence[Mapping[str, Any]]) -> dict[int, dict[int, float]]:
    matrix: dict[int, dict[int, float]] = {}
    for cell in cells:
        row = int(cell["after_task"])
        column = int(cell["eval_task"])
        if column > row:
            raise ValueError("CL matrix cannot contain future-task evaluation")
        if column in matrix.setdefault(row, {}):
            raise ValueError(f"Duplicate CL matrix cell R[{row},{column}]")
        matrix[row][column] = float(cell["accuracy"])
    expected = {(1, 1), (2, 1), (2, 2)}
    observed = {
        (row, column)
        for row, columns in matrix.items()
        for column in columns
    }
    if observed != expected:
        raise ValueError(f"Two-task matrix cells mismatch: {sorted(observed)}")
    return matrix


def build_cl_metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = matrix_rows(cells)
    r11 = matrix[1][1]
    r21 = matrix[2][1]
    r22 = matrix[2][2]
    if any(not 0.0 <= value <= 1.0 for value in (r11, r21, r22)):
        raise ValueError("CL accuracy values must use the 0..1 scale")
    return {
        "scale": "0..1",
        "average_accuracy_after_each_task": {
            "1": r11,
            "2": (r21 + r22) / 2.0,
        },
        "final_average_accuracy": (r21 + r22) / 2.0,
        "forgetting_task1": r11 - r21,
        "BWT": r21 - r11,
    }
