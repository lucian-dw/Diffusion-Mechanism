"""Real SiT pilot: paired Heun refinement and finite-amplitude defect response."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from _bootstrap import add_src_to_path

add_src_to_path()


def heun(velocity, x, labels, start, end, steps, keep=False):
    dt = (end - start) / steps
    history = [x] if keep else None
    for i in range(steps):
        a = velocity(x, start + i * dt, labels)
        b = velocity(x + dt * a, start + (i + 1) * dt, labels)
        x = x + (dt / 2) * (a + b)
        if history is not None:
            history.append(x)
    return torch.stack(history) if history is not None else x


def rms(x):
    return x.float().flatten(1).square().mean(dim=1).sqrt()


def paired_endpoints(velocity, anchor, defect, labels, amplitudes, steps):
    n = len(anchor)
    scales = anchor.new_tensor(amplitudes).view(-1, *([1] * anchor.ndim))
    x = (anchor[None] + scales * defect[None]).flatten(0, 1)
    out = heun(velocity, x, labels.repeat(len(amplitudes)), 0.5, 1.0, steps)
    return dict(zip(amplitudes, out.reshape(len(amplitudes), n, *anchor.shape[1:])))


def metrics(ref_low, ref_high, coarse, low, high):
    result = {"reference_gap": rms(ref_high - ref_low), "coarse_gap": rms(coarse - ref_high),
              "replay_base_gap": rms(high[0.0] - ref_high)}
    q_big = (high[0.01] - high[-0.01]) / 0.02
    q_small = (high[0.005] - high[-0.005]) / 0.01
    result["fd_step_gap_ratio"] = rms(q_big - q_small) / rms(q_small).clamp_min(1e-12)
    for a in (0.01, 0.1, 1.0):
        response = high[a] - high[0.0]
        low_response = low[a] - low[0.0]
        signal = rms(response)
        result[f"signal_{a:g}"] = signal
        result[f"paired_refinement_ratio_{a:g}"] = rms(response - low_response) / signal.clamp_min(1e-12)
        result[f"linear_error_ratio_{a:g}"] = rms(response - a * q_small) / signal.clamp_min(1e-12)
    result["reference_over_small_signal"] = result["reference_gap"] / result["signal_0.01"].clamp_min(1e-12)
    return result


def summarize(root):
    files = sorted(root.glob("shard-*/result.json"))
    if not files:
        raise SystemExit("No completed shards")
    shards = [json.loads(f.read_text()) for f in files]
    rows = [p for s in shards for p in s["paths"]]
    ids = sorted(p["id"] for p in rows)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate path IDs")
    summary = {"path_ids": ids, "paths": len(ids), "shards": len(shards),
               "gpu_seconds_per_path": sum(s["compute_seconds"] for s in shards) / len(ids),
               "load_seconds_by_shard": [s["load_seconds"] for s in shards],
               "phase_gpu_seconds_per_path": {key: sum(s["seconds"][key] for s in shards) / len(ids)
                                               for key in shards[0]["seconds"]},
               "max_peak_gib": max(s["peak_gib"] for s in shards)}
    for name in ("latent", "decoded"):
        summary[name] = {}
        for key in rows[0][name]:
            x = np.array([p[name][key] for p in rows])
            summary[name][key] = {"median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)),
                                  "max": float(x.max()), "under_0.1": int((x < 0.1).sum())}
    print(json.dumps(summary, indent=2, allow_nan=False))


@torch.inference_mode()
def run(args):
    from dm.imagenet256_flow import load_autoencoder_kl, load_sit_xl_2

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    started = time.perf_counter()
    model = load_sit_xl_2(args.checkpoint, device)
    vae = load_autoencoder_kl(args.vae, device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    ids = list(range(args.path_start, args.path_start + args.paths))
    seeds = [args.seed + i for i in ids]
    generators = [torch.Generator().manual_seed(s) for s in seeds]
    x = torch.stack([torch.randn((4, 32, 32), generator=g) for g in generators]).to(device)
    labels = torch.tensor([int(torch.randint(1000, (1,), generator=g)) for g in generators], device=device)
    phases = {}

    def timed(name, operation):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        phases[name] = time.perf_counter() - tick
        print(f"{name}: {phases[name]:.2f}s for {args.paths} paths", flush=True)
        return value

    def velocity(z, t, y):
        return torch.cat([model(zp, zp.new_full((len(zp),), t), yp)
                          for zp, yp in zip(z.split(args.forward_batch), y.split(args.forward_batch))])

    n = args.ref_steps
    # Keep the readout smooth: no clamp/uint8 conversion before differencing.
    def decode(z):
        return torch.cat([vae.decode(p / 0.18215).sample.float() for p in z.split(4)])

    print(f"paths={ids} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')} ref={n}/{2*n} tf32={args.tf32}", flush=True)
    velocity(x, 0.0, labels)
    ref_low = timed("reference_low", lambda: heun(velocity, x, labels, 0, 1, n, True))
    ref_high = timed("reference_high", lambda: heun(velocity, x, labels, 0, 1, 2*n, True))
    coarse = timed("heun_nfe8", lambda: heun(velocity, x, labels, 0, 1, 4, True))
    anchor = ref_high[n]
    defect = timed("defect", lambda: heun(velocity, ref_high[n // 2], labels, 0.25, 0.5, 1) - anchor)
    low = timed("suffix_low", lambda: paired_endpoints(velocity, anchor, defect, labels, (0., 0.01, 0.1, 1.), n // 2))
    high = timed("suffix_high", lambda: paired_endpoints(velocity, anchor, defect, labels, (0., 0.01, 0.1, 1., -0.01, -0.005, 0.005), n))
    a, b, c, low_img, high_img = timed("decode", lambda: (decode(ref_low[-1]), decode(ref_high[-1]), decode(coarse[-1]),
                                                          {k: decode(z) for k, z in low.items()},
                                                          {k: decode(z) for k, z in high.items()}))
    latent = metrics(ref_low[-1], ref_high[-1], coarse[-1], low, high)
    decoded = metrics(a, b, c, low_img, high_img)
    paths = [dict(id=i, seed=s, label=int(labels[j]), defect_rms=float(rms(defect)[j]),
                  latent={k: float(v[j]) for k, v in latent.items()},
                  decoded={k: float(v[j]) for k, v in decoded.items()})
             for j, (i, s) in enumerate(zip(ids, seeds))]
    report = dict(paths=paths, load_seconds=load_seconds, seconds=phases,
                  compute_seconds=sum(phases.values()), peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), ref_macro_steps=[n, 2*n], coarse_nfe=8,
                  injection_time=0.5, cfg=1, tf32=args.tf32)
    payload = {"reference_low": ref_low, "reference_high": ref_high, "coarse": coarse, "defect": defect}
    payload.update({f"suffix_high_{k:g}": v for k, v in high.items()})
    payload.update({f"suffix_low_{k:g}": v for k, v in low.items()})
    np.savez(out / "trajectories.npz", ids=ids, seeds=seeds, labels=labels.cpu().numpy(),
             **{k: v.cpu().numpy() for k, v in payload.items()})
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"completed paths={ids} compute={sum(phases.values()):.2f}s peak={report['peak_gib']:.2f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/SiT-XL-2-256x256.pt"))
    parser.add_argument("--vae", type=Path, default=Path("data/checkpoints/sd-vae-ft-mse"))
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--path-start", type=int, default=0)
    parser.add_argument("--paths", type=int, default=8)
    parser.add_argument("--ref-steps", type=int, default=64)
    parser.add_argument("--forward-batch", type=int, default=56)
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_dir)
    else:
        if args.ref_steps < 4 or args.ref_steps % 4 or args.paths < 1 or args.forward_batch < 1:
            parser.error("positive path/batch counts and reference steps divisible by four required")
        run(args)


if __name__ == "__main__":
    main()
