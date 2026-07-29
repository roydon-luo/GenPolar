"""Export minimal GenPolar release checkpoints from training checkpoints.

The exported checkpoints intentionally exclude optimizer state, fake-model state,
training arguments, epoch counters, and duplicate state dictionaries.  They keep
the key names expected by ``inference.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

REQUIRED_STATE_DICT_KEYS = ("unet_state_dict", "controlnet_state_dict")


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dictionary checkpoint, got {type(checkpoint)!r}: {path}"
        )
    return checkpoint


def require_state_dict(checkpoint: dict[str, Any], key: str) -> dict[str, torch.Tensor]:
    value = checkpoint.get(key)
    if not isinstance(value, dict) or not value:
        raise KeyError(f"Checkpoint does not contain a non-empty {key!r}")
    if not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    ):
        raise TypeError(f"{key!r} is not a plain tensor state dictionary")
    return value


def tensor_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def summarize(checkpoint: dict[str, Any]) -> None:
    print(f"top-level keys: {list(checkpoint)}")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            tensors = [
                item for item in value.values() if isinstance(item, torch.Tensor)
            ]
            if tensors:
                size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
                dtypes = sorted({str(tensor.dtype) for tensor in tensors})
                print(
                    f"  {key}: {len(tensors)} tensors, "
                    f"{size / 2**30:.3f} GiB, dtypes={dtypes}"
                )


def export_teacher(source: Path, output: Path) -> None:
    checkpoint = load_checkpoint(source)
    payload = {
        "format_version": 2,
        "model_role": "stage1_teacher",
        "prediction_type": checkpoint.get("prediction_type", "epsilon"),
        "stokes_scale": float(checkpoint.get("stokes_scale", 5.0)),
        "unet_state_dict": require_state_dict(checkpoint, "unet_state_dict"),
        "controlnet_state_dict": require_state_dict(
            checkpoint, "controlnet_state_dict"
        ),
    }
    save_payload(payload, output)


def export_one_step(source: Path, output: Path) -> None:
    checkpoint = load_checkpoint(source)
    student_key = (
        "student_unet_state_dict"
        if "student_unet_state_dict" in checkpoint
        else "unet_state_dict"
    )
    payload = {
        "format_version": 2,
        "model_role": "one_step_inference",
        "prediction_type": checkpoint.get("prediction_type", "epsilon"),
        "stokes_scale": float(checkpoint.get("stokes_scale", 5.0)),
        "unet_state_dict": require_state_dict(checkpoint, student_key),
        "controlnet_state_dict": require_state_dict(
            checkpoint, "controlnet_state_dict"
        ),
    }
    save_payload(payload, output)


def save_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output)
    size = sum(
        tensor_size_bytes(value)
        for value in payload.values()
        if isinstance(value, dict)
    )
    print(f"saved: {output}")
    print(f"state-dict tensor size: {size / 2**30:.3f} GiB")
    print(f"file size: {output.stat().st_size / 2**30:.3f} GiB")


def verify_export(role: str, source: Path, exported: Path) -> None:
    source_checkpoint = load_checkpoint(source)
    exported_checkpoint = load_checkpoint(exported)

    expected_top_level = {
        "format_version",
        "model_role",
        "prediction_type",
        "stokes_scale",
        "unet_state_dict",
        "controlnet_state_dict",
    }
    if set(exported_checkpoint) != expected_top_level:
        raise ValueError(
            f"Unexpected exported keys: {set(exported_checkpoint)}; "
            f"expected: {expected_top_level}"
        )

    source_unet_key = (
        "student_unet_state_dict"
        if role == "one-step" and "student_unet_state_dict" in source_checkpoint
        else "unet_state_dict"
    )
    comparisons = (
        (source_unet_key, "unet_state_dict"),
        ("controlnet_state_dict", "controlnet_state_dict"),
    )
    for source_key, exported_key in comparisons:
        source_state = require_state_dict(source_checkpoint, source_key)
        exported_state = require_state_dict(exported_checkpoint, exported_key)
        if source_state.keys() != exported_state.keys():
            raise ValueError(f"Parameter names differ for {exported_key}")
        for name, source_tensor in source_state.items():
            exported_tensor = exported_state[name]
            if source_tensor.shape != exported_tensor.shape:
                raise ValueError(f"Shape differs for {exported_key}.{name}")
            if source_tensor.dtype != exported_tensor.dtype:
                raise ValueError(f"Dtype differs for {exported_key}.{name}")
            if not torch.equal(source_tensor, exported_tensor):
                raise ValueError(f"Tensor values differ for {exported_key}.{name}")
        print(f"verified: {exported_key} ({len(exported_state)} tensors)")

    print(f"export verification passed: {exported}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Summarize a training checkpoint"
    )
    inspect_parser.add_argument("checkpoint", type=Path)

    for command in ("teacher", "one-step"):
        export_parser = subparsers.add_parser(
            command, help=f"Export a minimal {command} checkpoint"
        )
        export_parser.add_argument("source", type=Path)
        export_parser.add_argument("output", type=Path)

    verify_parser = subparsers.add_parser(
        "verify", help="Verify an exported checkpoint"
    )
    verify_parser.add_argument("role", choices=("teacher", "one-step"))
    verify_parser.add_argument("source", type=Path)
    verify_parser.add_argument("exported", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "inspect":
        summarize(load_checkpoint(args.checkpoint))
    elif args.command == "teacher":
        export_teacher(args.source, args.output)
    elif args.command == "one-step":
        export_one_step(args.source, args.output)
    elif args.command == "verify":
        verify_export(args.role, args.source, args.exported)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
