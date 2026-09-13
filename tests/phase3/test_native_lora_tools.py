from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "phase3" / "native_lora_tools.py"
SPEC = importlib.util.spec_from_file_location("native_lora_tools", MODULE_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def expected_names():
    return [
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in range(36)
        for projection in ("q_proj", "v_proj")
    ]


def test_native_lora_target_filter():
    names = expected_names() + ["model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1"]
    targets = TOOLS.language_qv_targets(names)
    assert len(targets) == 72
    assert TOOLS.classify_names(targets) == {
        "total": 72, "language": 72, "q_proj": 36, "v_proj": 36, "vision": 0, "merger_aligner": 0
    }


def test_native_lora_trainable_parameters():
    lora = [f"base_model.{name}.lora_A.default.weight" for name in expected_names()]
    assert TOOLS.classify_names(lora)["vision"] == 0
    assert all(".lora_A." in name or ".lora_B." in name for name in lora)


def test_native_lora_state_dict():
    state = {}
    for name in expected_names():
        state[f"base_model.{name}.lora_A.weight"] = [48, 4096]
        state[f"base_model.{name}.lora_B.weight"] = [4096, 48]
    audit = TOOLS.audit_state_keys(state, rank=48)
    assert audit["status"] == "PASS"
    assert audit["wrapper_count"] == 72


def test_native_lora_reload():
    assert TOOLS.max_abs_difference([0.0, 1.0], [0.0, 1.0]) == 0.0
    assert TOOLS.max_abs_difference([0.0, 1.0], [0.0, 1.5]) == 0.5


def test_native_lora_logits_difference():
    delta = TOOLS.max_abs_difference([1.0, -2.0, 3.0], [1.0, -1.75, 3.0])
    assert delta > 0


def test_vqa_output_parser():
    prompt = "Question\nOptions:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta"
    parsed = TOOLS.parse_vqa_output("C. Gamma", prompt, "Gamma")
    assert parsed["raw_output"] == "C. Gamma"
    assert parsed["valid_parse"]
    assert parsed["option_letter"] == "C"
    assert parsed["answer_text"] == "Gamma"
    assert parsed["correct"]
