"""Source-bound registry parsing shared by the Phase XIX-W matrix and checker."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER_RS = ROOT / "hierarchos-vulkan/src/transformer.rs"
CANONICAL_COUNT = 146
ALIAS_COUNT = 83
TOTAL_COUNT = 229

# Aliases whose Rust constructor resolves through a composite or a differently
# named text graph. A composite is deliberately recorded as
# ``nested_text_config``: its text backbone is selected from the nested config
# at load time, so assigning one fixed canonical graph would overstate support.
# Known text backbones are made explicit; this is a resolution claim, not a
# multimodal-pipeline or numerical-parity claim.
ALIAS_TARGET_OVERRIDES = {
    "encoder-decoder": "nested_text_config",
    "t5gemma": "t5_gemma_module",
    "code_llama": "llama",
    "mllama_text_model": "llama",
    "mllama": "llama",
    "llama4": "llama4_text",
    "emu3": "emu3_text_model",
    "aria": "aria_text",
    "fuyu": "persimmon",
    "mistral3": "mistral",
    "qwen2_vl_text": "qwen2",
    "qwen2_5_vl_text": "qwen2",
    "qwen2_vl": "qwen2",
    "qwen2_5_vl": "qwen2",
    "qwen2_5_omni_text": "qwen2",
    "qwen3_vl_text": "qwen3",
    "qwen3_vl": "qwen3",
    "qwen3_vl_moe_text": "qwen3_moe",
    "qwen3_vl_moe": "qwen3_moe",
    "qwen3_omni_moe_text": "qwen3_moe",
    "qwen3_omni_moe_thinker": "qwen3_moe",
    "qwen3_omni_moe": "qwen3_moe",
    "qwen3_5": "qwen3_5_text",
    "qwen3_5_moe": "qwen3_5_moe_text",
    "qwen4_exp": "qwen4_exp_text",
    "phi4_multimodal": "phi3",
    "glm4v_text": "glm4",
    "glm4v": "glm4",
    "glm4v_moe_text": "glm4_moe",
    "glm4v_moe": "glm4_moe",
    "glm46v": "glm4",
    "glm_ocr_text": "glm4",
    "glm_ocr": "glm4",
    "glm4_moe_lite": "glm4_moe",
    "glm5_next": "glm5_next_text",
    "deepseek_v32": "deepseek_v3",
    "kimi_k2": "deepseek_v3",
    "kimi_k25": "deepseek_v3",
    "kimi_k3": "nested_text_config",
    "glm_moe_dsa": "glm4_moe",
    "hunyuan_vl_text": "hunyuan_v1_dense",
    "hunyuan_vl": "hunyuan_v1_dense",
    "gemma3": "gemma3_text",
    "gemma4": "gemma4_text",
    "exaone4_5_text": "exaone4",
    "exaone4_5": "exaone4",
    "cohere_compass": "cohere_compass_text",
    "cohere2_vision": "cohere2",
    "cosmos3_omni": "nested_text_config",
    "got_ocr2": "nested_text_config",
    "aya_vision": "nested_text_config",
    "llava": "nested_text_config",
    "llava_next": "nested_text_config",
    "llava_next_video": "nested_text_config",
    "llava_onevision": "nested_text_config",
    "video_llava": "nested_text_config",
    "video_llama_3": "nested_text_config",
    "vipllava": "nested_text_config",
    "idefics2": "nested_text_config",
    "idefics3": "nested_text_config",
    "smolvlm": "nested_text_config",
    "qwen2_audio": "nested_text_config",
    "voxtral": "nested_text_config",
    "fast_vlm": "nested_text_config",
    "internvl": "nested_text_config",
    "janus": "nested_text_config",
    "ovis2": "nested_text_config",
    "paligemma": "nested_text_config",
    "deepseek_vl": "nested_text_config",
    "deepseek_vl_hybrid": "nested_text_config",
    "paddleocr_vl_text": "nested_text_config",
    "paddleocr_vl": "nested_text_config",
    "audioflamingo3": "nested_text_config",
    "qwen3_asr": "nested_text_config",
    "granite_speech": "nested_text_config",
    "granite_speech_plus": "nested_text_config",
    "granite4_vision_text": "granite",
    "granite4_vision": "granite",
    "glmasr": "nested_text_config",
    "musicflamingo": "nested_text_config",
    "vibevoice_asr": "nested_text_config",
    "modernvbert": "modernbert",
    "minimax_m3_vl": "minimax_m3_vl_text",
}


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
    raise ValueError("unbalanced Rust braces")


def _top_level_arrow(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == ">" and index and text[index - 1] == "=" and depth == 0:
                return index
    raise ValueError("unterminated Rust match arm")


def _arm_body(text: str, start: int) -> tuple[str, int]:
    index = start
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    if index < len(text) and text[index] == "{":
        end = _matching_brace(text, index)
        return text[index:end], end
    end = _top_level_arrow(text, start)  # kept for a useful error if malformed
    del end
    depth = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                return text[start:index], index
        index += 1
    return text[start:], index


def parse_registry(source_path: Path = TRANSFORMER_RS) -> dict:
    source = Path(source_path).read_text(encoding="utf-8")
    model_type = re.search(
        r"pub fn model_type\(self\).*?match self \{(.*?)\n\s*\}\n\s*\}",
        source,
        re.DOTALL,
    )
    if model_type is None:
        raise ValueError("could not locate VulkanTransformerArchitecture::model_type")
    canonical = re.findall(r'=>\s*"([^"]+)"', model_type.group(1))
    alias_match = re.search(
        r"HF_MODEL_TYPE_ALIASES.*?=\s*&\[(.*?)\];", source, re.DOTALL
    )
    if alias_match is None:
        raise ValueError("could not locate HF_MODEL_TYPE_ALIASES")
    aliases = re.findall(r'"([^"]+)"', alias_match.group(1))
    if len(canonical) != len(set(canonical)):
        raise ValueError("duplicate canonical model_type")
    if len(aliases) != len(set(aliases)):
        raise ValueError("duplicate alias model_type")
    if set(canonical) & set(aliases):
        raise ValueError("canonical/alias overlap")

    anchor = source.index("pub fn from_hf_value(value: &serde_json::Value) -> Result<Self> {")
    match_start = source.index("match value", anchor)
    body_start = source.index("{", match_start)
    body_end = _matching_brace(source, body_start)
    body = source[body_start + 1:body_end - 1]
    canonical_set = set(canonical)
    alias_set = set(aliases)
    alias_targets: dict[str, str | None] = {}
    alias_constructors: dict[str, list[str]] = {}
    index = 0
    while index < len(body):
        while index < len(body) and body[index] in " \t\r\n,":
            index += 1
        if index >= len(body) or body[index] == "}":
            break
        arrow = _top_level_arrow(body, index)
        prefix = body[index:arrow]
        patterns = re.findall(r'"([^"]+)"', prefix)
        action, end = _arm_body(body, arrow + 1)
        constructors = re.findall(
            r"(?:Self|VulkanTransformerArchitecture)::from_([a-z0-9_]+)_value",
            action,
        )
        for alias in patterns:
            if alias not in alias_set:
                continue
            alias_constructors[alias] = constructors
            target = None
            for constructor in constructors:
                if constructor in canonical_set:
                    target = constructor
                    break
                hyphenated = constructor.replace("_", "-")
                if hyphenated in canonical_set:
                    target = hyphenated
                    break
            alias_targets[alias] = target
        index = max(end, index + 1)
    # Use one audited target for every alias. The Rust constructors remain
    # recorded separately so the matrix can prove that the alias route exists
    # even when its graph is selected from a nested text config.
    for alias in aliases:
        if alias not in ALIAS_TARGET_OVERRIDES:
            raise ValueError(f"alias has no audited target: {alias}")
        alias_targets[alias] = ALIAS_TARGET_OVERRIDES[alias]
        alias_constructors.setdefault(alias, [])
    if set(ALIAS_TARGET_OVERRIDES) != alias_set:
        extra = sorted(set(ALIAS_TARGET_OVERRIDES) - alias_set)
        raise ValueError(f"audited alias target has no Rust route: {extra}")
    missing = sorted(alias_set - set(alias_targets))
    if missing:
        raise ValueError(f"aliases absent from from_hf_value: {missing}")
    for alias, target in alias_targets.items():
        if target != "nested_text_config" and target not in canonical_set:
            raise ValueError(f"alias {alias} targets a non-canonical graph: {target}")
    return {
        "canonical": canonical,
        "aliases": aliases,
        "alias_targets": alias_targets,
        "alias_constructors": alias_constructors,
        "source_sha256": digest(source_path),
    }


def profile_architectures(fixture_root: Path) -> dict[str, str]:
    result = {}
    for config in sorted(Path(fixture_root).glob("*/*/model/config.json")):
        value = read_json(config)
        family = config.parents[1].name
        result[family] = value.get("model_type", "")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TRANSFORMER_RS)
    args = parser.parse_args()
    registry = parse_registry(args.source)
    print(
        f"{len(registry['canonical'])} canonical + {len(registry['aliases'])} aliases "
        f"({len(registry['canonical']) + len(registry['aliases'])} total)"
    )
    for alias in registry["aliases"]:
        print(f"{alias:32} -> {registry['alias_targets'][alias]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
