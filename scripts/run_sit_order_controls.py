"""Matched matrix-free order-blind and isotropic controls for saved SiT paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from run_sit_rollout_calibration import rms


def exp_action(matvec, x, order, checkpoints):
    """Independent twice-reorthogonalized Arnoldi projections for a batch."""
    norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    basis = [x / norm.clamp_min(1e-20)]
    hess = x.new_zeros((len(x), order + 1, order))
    outputs = {}
    for j in range(order):
        v = matvec(basis[-1])
        q = torch.stack(basis, dim=2)
        for _ in range(2):
            coeff = torch.bmm(q.transpose(1, 2), v.unsqueeze(2)).squeeze(2)
            hess[:, :j+1, j] += coeff
            v = v - torch.bmm(q, coeff.unsqueeze(2)).squeeze(2)
        beta = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        hess[:, j+1, j] = beta[:, 0]
        if j + 1 in checkpoints:
            y = torch.matrix_exp(hess[:, :j+1, :j+1].double())[:, :, 0].to(x.dtype)
            outputs[j + 1] = torch.bmm(q, y.unsqueeze(2)).squeeze(2) * norm
        basis.append(torch.where(beta > 1e-7, v / beta.clamp_min(1e-20), 0))
    return outputs


def simpson_weights(intervals):
    if intervals < 2 or intervals % 2:
        raise ValueError("Simpson quadrature requires an even number of intervals")
    w = np.ones(intervals + 1)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return w / (6 * intervals)  # duration .5


def isotropic(defect, ids, count):
    signs = [torch.randint(2, (count, *defect.shape[1:]), generator=torch.Generator().manual_seed(311172 + int(p)))
             for p in ids]
    return (torch.stack(signs).to(defect) * 2 - 1) * rms(defect).view(-1, 1, 1, 1, 1)


def angle(a, b):
    a, b = a.flatten(1), b.flatten(1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-20)
    return cos.clamp(-1, 1).acos().rad2deg()


def ratio(a, b):
    return rms(a - b) / rms(b).clamp_min(1e-12)


@torch.inference_mode()
def run(args):
    from dm.imagenet256_flow import load_autoencoder_kl, load_sit_xl_2

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    with np.load(args.input, allow_pickle=False) as d:
        idx = slice(args.offset, args.offset + args.paths if args.paths else None)
        ids, labels = d["ids"][idx].tolist(), torch.as_tensor(d["labels"][idx], device=device)
        defect = torch.as_tensor(d["defect"][idx], device=device)
        ref = torch.as_tensor(d["reference_high"][:, idx], device=device)
        z = {a: torch.as_tensor(d[f"suffix_high_{a:g}"][idx], device=device)
             for a in (0., 0.01, 0.1, 1., -0.005, 0.005)}
    assert len(ids) and len(ref) % 2 == 1
    phases = {}

    def timed(key, call):
        torch.cuda.synchronize()
        t = time.perf_counter()
        value = call()
        torch.cuda.synchronize()
        phases[key] = time.perf_counter() - t
        print(f"{key}: {phases[key]:.2f}s paths={ids}", flush=True)
        return value

    model, vae = timed("load", lambda: (load_sit_xl_2(args.checkpoint, device), load_autoencoder_kl(args.vae, device)))
    starts = torch.cat([defect[:, None], isotropic(defect, ids, args.probes)], dim=1)
    n, k = starts.shape[:2]
    shape = defect.shape[1:]

    def solve(vectors, intervals, eps, checks):
        width = vectors.shape[1]
        half = (len(ref) - 1) // 2
        if half % intervals:
            raise ValueError("Quadrature nodes must be on the stored reference grid")
        anchors = ref[half::half // intervals].repeat_interleave(width, dim=1)
        w = ref.new_tensor(simpson_weights(intervals))
        time_nodes = torch.linspace(0.5, 1, intervals + 1, device=device)
        ts = time_nodes[:, None].expand(-1, n * width).flatten()
        ys = labels.repeat_interleave(width).repeat(intervals + 1)

        def matvec(v):
            u = v.reshape(n * width, *shape)
            scale = rms(u).clamp_min(1e-20)
            step = (eps / scale).view(-1, 1, 1, 1) * u
            pair = torch.stack([anchors + step, anchors - step]).flatten(0, 2)
            field = torch.cat([model(x, t, y) for x, t, y in
                               zip(pair.split(args.batch), ts.repeat(2).split(args.batch), ys.repeat(2).split(args.batch))])
            plus, minus = field.reshape(2, intervals + 1, n * width, *shape)
            total = torch.einsum("t,tbchw->bchw", w, plus - minus)
            return (total * (scale / (2 * eps)).view(-1, 1, 1, 1)).flatten(1)

        return {i: q.reshape(n, width, *shape) for i, q in exp_action(
            matvec, vectors.flatten(0, 1).flatten(1), args.krylov, checks).items()}

    k0 = max(1, args.krylov - 3)
    predicted = timed("exponential", lambda: solve(starts, args.quad, args.eps, (k0, args.krylov)))
    ordered = (z[0.005] - z[-0.005]) / 0.01
    main = predicted[args.krylov]

    def decode(x):
        return torch.cat([vae.decode(v / 0.18215).sample.float() for v in x.split(args.decode_batch)])

    def readout(q, eps=0.002):
        width = q.shape[1]
        q = q.flatten(0, 1)
        baseline = z[0.].repeat_interleave(width, 0)
        r = rms(q).clamp_min(1e-20)
        dq = q * (eps / r).view(-1, 1, 1, 1)
        im = decode(torch.cat([baseline + dq, baseline - dq]))
        a, b = im.chunk(2)
        y = (a - b) * (r / (2 * eps)).view(-1, 1, 1, 1)
        return y.reshape(n, width, *y.shape[1:])

    qa, qb, ordered_img, images = timed("readout", lambda: (
        readout(main), readout(predicted[k0]), readout(ordered[:, None])[:, 0],
        {a: decode(x) for a, x in z.items()}))
    checks = {"decode_ordered_fd": ratio(ordered_img, (images[0.005] - images[-0.005]) / 0.01)}
    for kind, a, b in [("latent", predicted[k0], main), ("decoded", qb, qa)]:
        checks[f"krylov_{kind}"] = ratio(a.flatten(0, 1), b.flatten(0, 1)).reshape(n, k).amax(1)
    if args.validate:
        for key, intervals, eps in (("quad", args.quad * 2, args.eps), ("fd", args.quad, args.eps / 2)):
            refined = timed(key, lambda: solve(starts[:, :2], intervals, eps, (args.krylov,))[args.krylov])
            decoded = timed(key + "_readout", lambda: readout(refined))
            for kind, a, b in [("latent", main[:, :2], refined), ("decoded", qa[:, :2], decoded)]:
                r = ratio(a.flatten(0, 1), b.flatten(0, 1)).reshape(n, 2)
                checks[f"{key}_{kind}_real"], checks[f"{key}_{kind}_probe"] = r.unbind(1)

    def measure(qo, q0, actual):
        data = {"ordered_energy": rms(qo).square(), "blind_energy": rms(q0[:, 0]).square(),
                "iso_energies": rms(q0[:, 1:].flatten(0, 1)).square().reshape(n, -1),
                "order_vector_delta_energy": rms(qo - q0[:, 0]).square(), "angle_deg": angle(qo, q0[:, 0])}
        for a in (0.01, 0.1, 1.0):
            y = actual[a] - actual[0.]
            data[f"actual_{a:g}"] = rms(y).square()
            data[f"ord_sqerr_{a:g}"] = rms(a * qo - y).square()
            data[f"blind_sqerr_{a:g}"] = rms(a * q0[:, 0] - y).square()
        return {key: value.cpu().tolist() for key, value in data.items()}

    report = {"ids": ids, "seconds": phases, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
              "checks": {key: v.cpu().tolist() for key, v in checks.items()},
              "latent": measure(ordered, main, z), "decoded": measure(ordered_img, qa, images)}
    np.savez(out / "latent_controls.npz", ids=ids, ordered=ordered.cpu().numpy(),
             blind=main[:, 0].cpu().numpy(), isotropic=main[:, 1:].cpu().numpy())
    temporary = out / "controls.partial.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.rename(out / "controls.json")
    print(f"completed paths={ids} peak={report['peak_gib']:.2f} GiB", flush=True)


def summarize(paths):
    reports = [json.loads(p.read_text()) for p in paths]
    ids = np.concatenate([r["ids"] for r in reports])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate path IDs")
    summary = {"ids": sorted(ids.tolist()), "groups": {}}
    summary["cost_seconds_per_path"] = {
        key: sum(r["seconds"][key] for r in reports if key in r["seconds"]) /
        sum(len(r["ids"]) for r in reports if key in r["seconds"])
        for key in sorted(set().union(*(r["seconds"] for r in reports)))}
    summary["checks_pct"] = {key: (100 * np.quantile(np.concatenate([
        r["checks"][key] for r in reports if key in r["checks"]]), [0.5, 0.9, 1.0])).tolist()
        for key in sorted(set().union(*(r["checks"] for r in reports)))}
    rng = np.random.default_rng(144240)
    for key, keep in (("pilot", ids < 16), ("new", ids >= 16), ("combined", np.ones(len(ids), bool))):
        if not keep.any():
            continue
        n = int(keep.sum())
        ix = rng.integers(n, size=(2000, n))
        group = summary["groups"][key] = {"n": n}
        for kind in ("latent", "decoded"):
            def arr(name):
                return np.concatenate([r[kind][name] for r in reports])[keep]

            o, b, probes = arr("ordered_energy"), arr("blind_energy"), arr("iso_energies")
            iso = probes.mean(1)
            m = probes.shape[1]
            sampled = probes[ix[:, :, None], rng.integers(m, size=(len(ix), n, m))].mean(2)
            row = group[kind] = dict(order_energy_gap_pct=100 * np.abs(o-b).sum()/o.sum(),
                                     order_vector_gap_pct=100 * np.sqrt(arr("order_vector_delta_energy").sum()/o.sum()),
                                     angle_deg=np.quantile(arr("angle_deg"), [0.5, 0.9]).tolist(),
                                     iso_mc_se_rel_median_pct=100 * np.median(probes.std(1, ddof=1)/np.sqrt(m)/iso),
                                     iso_split_gap_pct=100 * np.abs(probes[:, :m//2].mean(1)-probes[:, m//2:].mean(1)).sum()/iso.sum(),
                                     amplitudes={})
            for a in (0.01, 0.1, 1.):
                s = f"{a:g}"
                target = arr(f"actual_{s}") / (a*a)
                eo, eb, ei = [np.abs(y-target) for y in (o, b, iso)]
                denom = target[ix].sum(1)
                direction = 100*(np.abs(sampled-target[ix]).sum(1)-eb[ix].sum(1))/denom
                order = 100*(eb-eo)[ix].sum(1)/denom
                row["amplitudes"][s] = dict(energy_error_pct=[100*e.sum()/target.sum() for e in (ei, eb, eo)],
                    direction_gain_pp=100*(ei-eb).sum()/target.sum(), order_gain_pp=100*(eb-eo).sum()/target.sum(),
                    direction_ci_pp=np.quantile(direction,[0.025,0.975]).tolist(), order_ci_pp=np.quantile(order,[0.025,0.975]).tolist(),
                    vector_error_pct=[100*np.sqrt(arr(f"{p}_sqerr_{s}").sum()/arr(f"actual_{s}").sum()) for p in ("blind", "ord")])
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", nargs="+", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--paths", type=int)
    parser.add_argument("--quad", type=int, default=8)
    parser.add_argument("--krylov", type=int, default=9)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--eps", type=float, default=0.003)
    parser.add_argument("--batch", type=int, default=56)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/SiT-XL-2-256x256.pt"))
    parser.add_argument("--vae", type=Path, default=Path("data/checkpoints/sd-vae-ft-mse"))
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif not args.input or not args.output or args.probes < 2 or args.krylov < 4 or args.eps <= 0:
        parser.error("Need input/output, >=2 probes, >=4 Krylov steps, and positive FD scale")
    else:
        run(args)


if __name__ == "__main__":
    main()
