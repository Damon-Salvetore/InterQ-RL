"""QuRL Update-Aware Quantization (UAQ): divide each RMSNorm-fed linear by s and multiply the norm by s.

Groups: input_layernorm -> q/k/v_proj, post_attention_layernorm -> gate/up_proj. On MoE blocks only the
attention group is rescaled.

Usage: python scripts/apply_uaq.py --src <hf checkpoint> --dst <output dir> [--verify]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

GROUPS = (
    ("input_layernorm", ("q_proj", "k_proj", "v_proj")),
    ("post_attention_layernorm", ("gate_proj", "up_proj")),
)


def rescale_(model: torch.nn.Module, s: float) -> dict[str, int]:
    """Apply W <- W/s and norm <- norm*s in place. Returns what was touched."""

    if not (s > 0):
        raise ValueError(f"scale must be positive, got {s}")
    touched = {"norms": 0, "linears": 0, "layers": 0, "moe_ffn_skipped": 0}
    layers = model.model.layers
    for layer in layers:
        hit = False
        for norm_name, linear_names in GROUPS:
            norm = getattr(layer, norm_name, None)
            if norm is None:
                continue
            if norm_name == "post_attention_layernorm" and _is_moe(layer):
                touched["moe_ffn_skipped"] += 1
                continue
            targets = []
            for ln in linear_names:
                targets.extend(_resolve_all(layer, ln))
            if not targets:
                continue
            with torch.no_grad():
                norm.weight.mul_(s)
                for mod in targets:
                    mod.weight.div_(s)
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.div_(s)
            touched["norms"] += 1
            touched["linears"] += len(targets)
            hit = True
        touched["layers"] += int(hit)
    return touched


def _is_moe(layer) -> bool:
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    return experts is not None and len(experts) > 0


def _router_weight(model, idx: int):
    return model.model.layers[idx].mlp.gate.weight


def _snapshot_routers(model) -> dict:
    out = {}
    for i, layer in enumerate(model.model.layers):
        if not _is_moe(layer):
            continue
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and hasattr(gate, "weight"):
            out[i] = gate.weight.detach().clone()
    return out


def _resolve_all(layer, name: str) -> list:
    attn = getattr(layer, "self_attn", None)
    if attn is not None and hasattr(attn, name):
        return [getattr(attn, name)]

    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return []
    if hasattr(mlp, name):
        return [getattr(mlp, name)]

    found = []
    for expert in getattr(mlp, "experts", None) or []:
        if hasattr(expert, name):
            found.append(getattr(expert, name))
    shared = getattr(mlp, "shared_expert", None)
    if shared is not None and hasattr(shared, name):
        found.append(getattr(shared, name))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF checkpoint to rescale")
    ap.add_argument("--dst", required=True, help="where to write the rescaled copy")
    ap.add_argument("--scale", type=float, default=1.5,
                    help="rescale factor s")
    ap.add_argument("--verify", action="store_true",
                    help="check the rescale is output-preserving before saving")
    ap.add_argument("--verify-tokens", type=int, default=64)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")

    print(f"loading {src}")
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float32, trust_remote_code=True, device_map="cpu")
    model.eval()

    ref = None
    if args.verify:
        ids = torch.arange(1, args.verify_tokens + 1).unsqueeze(0) % cfg.vocab_size
        with torch.no_grad():
            ref = model(ids).logits.clone()

    router_before = _snapshot_routers(model)
    touched = rescale_(model, args.scale)
    print(f"rescaled s={args.scale}: {touched['layers']} layers, "
          f"{touched['norms']} norms, {touched['linears']} linears, "
          f"{touched['moe_ffn_skipped']} MoE FFN groups skipped")
    if touched["linears"] == 0:
        raise SystemExit("no linear layer was rescaled")
    n_layers = len(model.model.layers)
    expected = (len(GROUPS) * n_layers) - touched["moe_ffn_skipped"]
    if touched["norms"] != expected:
        raise SystemExit(f"expected {expected} norms rescaled, got {touched['norms']}")
    changed = [n for n, w in router_before.items()
               if not torch.equal(_router_weight(model, n), w)]
    if changed:
        raise SystemExit(f"{len(changed)} router weights changed")

    if args.verify:
        with torch.no_grad():
            new = model(ids).logits
        delta = (new - ref).abs().max().item()
        rel = delta / ref.abs().max().item()
        print(f"verify: max|dlogit| = {delta:.3e}  relative = {rel:.3e}")
        if rel > 1e-4:
            raise SystemExit("rescale is not output-preserving")
        print("verify: OK, transformation is output-preserving")

    print(f"saving to {dst}")
    dst.mkdir(parents=True)
    model.to(torch.bfloat16).save_pretrained(dst)
    tok.save_pretrained(dst)
    for extra in ("generation_config.json",):
        p = src / extra
        if p.exists():
            shutil.copy2(p, dst / extra)
    (dst / "UAQ_README.json").write_text(json.dumps({
        "source": str(src),
        "scale": args.scale,
        "groups": [[n, list(ls)] for n, ls in GROUPS],
        "reference": "QuRL UAQ (Li et al., 2026)",
    }, indent=2))
    print("done")


if __name__ == "__main__":
    main()
