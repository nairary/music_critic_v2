#!/usr/bin/env python3
"""Synthetic-only index_add and H/member exact replay diagnostic.

Run each policy in a fresh process, with CUBLAS_WORKSPACE_CONFIG=:4096:8
provided by its launcher before Torch/CUDA initialization. Native execution
requires its own registered engineering gate. This reads no corpus/cache and
never writes model checkpoints. JSON hashes cover tensor contents, not pickle
or torch.save container bytes. No scientific-quality claim follows this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time


def tensor_content_hash(values):
    """Hash named dense tensor metadata and raw bytes without NumPy dependency."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(b'\n')
        # Reshape first so scalar tensors can also be viewed as raw bytes.
        raw = tensor.reshape(-1).view(torch.uint8)
        for chunk in raw.split(65536):
            digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def compare_tensors(reference, other):
    import torch
    if reference.keys() != other.keys():
        raise ValueError('comparison requires the same named tensors')
    differences = {}
    for name, left in reference.items():
        right = other[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('comparison requires identical shape/dtype')
        if not torch.equal(left, right):
            differences[name] = float((left.double() - right.double()).abs().max()) if left.numel() else 0.0
    return dict(exact_equal=not differences, changed_tensors=len(differences),
                max_abs_diff=max(differences.values(), default=0.0),
                differing_names=sorted(differences))


def _copy(values):
    return {name: tensor.detach().cpu().clone() for name, tensor in values.items()}


def _summarize(replays):
    return dict(content_sha256=[tensor_content_hash(values) for values in replays],
                comparisons_to_first=[compare_tensors(replays[0], values) for values in replays[1:]])


def _training_trace(fixture, kind, seed, steps, device):
    import torch
    from music_critic.experiments.harmonic_event_reasoner.model import EventReadout, masked_membership_loss
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = EventReadout(input_width=266, kind=kind, hidden=64, rounds=3,
                         node_types=9, edge_types=8).to(device)
    trace = dict(initial_weights=_copy(model.state_dict()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(fixture['x'], fixture['node_type'], fixture['edge_index'],
                       fixture['edge_type'], 0, fixture['candidates'])
        loss = masked_membership_loss(logits, fixture['labels'], fixture['mask'])
        if not torch.isfinite(loss):
            raise RuntimeError('synthetic probe produced nonfinite loss')
        loss.backward()
        losses.append(loss.detach().cpu().clone())
        if step == 0:
            trace['initial_logits'] = _copy(dict(logits=logits))
            trace['initial_gradients'] = _copy({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0:
            trace['weights_after_one_update'] = _copy(model.state_dict())
    trace['final_weights'] = _copy(model.state_dict())
    with torch.no_grad():
        trace['final_logits'] = _copy(dict(logits=model(fixture['x'], fixture['node_type'],
            fixture['edge_index'], fixture['edge_type'], 0, fixture['candidates'])))
    trace['losses'] = dict(losses=torch.stack(losses))
    return trace


def run_probe(*, device='cpu', deterministic=False, repeats=3, steps=8, nodes=96,
              edges=2048, reduction_edges=32768, kinds=('self', 'graph')):
    """Replay identical tensors and independently re-seeded synthetic training."""
    import torch
    from music_critic.experiments.harmonic_event_reasoner.reproducibility import deterministic_runtime
    if (type(repeats) is not int or repeats < 2 or type(steps) is not int or steps < 1
            or nodes < 4 or edges < 1 or reduction_edges < 1
            or not kinds or any(k not in {'mlp', 'transe', 'self', 'graph'} for k in kinds)):
        raise ValueError('positive synthetic dimensions, >=2 repeats and known kinds required')
    started = time.monotonic()
    with deterministic_runtime(device, enabled=deterministic) as runtime:
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('requested CUDA unavailable')
        generator = torch.Generator(device='cpu').manual_seed(731)
        fixture = dict(
            x=torch.randn(nodes, 266, generator=generator),
            node_type=torch.arange(nodes) % 9,
            edge_index=torch.stack((torch.randint(nodes, (edges,), generator=generator),
                torch.randint(max(2, nodes // 8), (edges,), generator=generator))),
            edge_type=torch.randint(8, (edges,), generator=generator),
            candidates=torch.arange(1, nodes),
            labels=(torch.arange(nodes - 1) % 2).float(),
            mask=torch.ones(nodes - 1, dtype=torch.bool),
        )
        fixture_hash = tensor_content_hash(fixture)
        fixture = {name: tensor.to(device) for name, tensor in fixture.items()}
        # Same source, receiver and base tensor objects for every reduction.
        # Repeated destinations and varied magnitudes expose addition ordering.
        source = torch.randn(reduction_edges, 64, generator=generator)
        source *= torch.logspace(-3, 3, reduction_edges).unsqueeze(1)
        receiver = torch.randint(17, (reduction_edges,), generator=generator)
        reduction_hash = tensor_content_hash(dict(source=source, receiver=receiver))
        source, receiver = source.to(device), receiver.to(device)
        base = torch.zeros(17, 64, device=device)
        reductions = [_copy(dict(aggregate=base.index_add(0, receiver, source))) for _ in range(repeats)]
        results = {}
        for kind in kinds:
            replays = [_training_trace(fixture, kind, 17, steps, device) for _ in range(repeats)]
            results[kind] = {stage: _summarize([replay[stage] for replay in replays]) for stage in replays[0]}
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        return dict(schema='HMemberDeterminismProbe@1', runtime=runtime,
            synthetic_only=True, corpus_accessed=False, checkpoints_accessed=False,
            fixture_sha256=fixture_hash, reduction_input_sha256=reduction_hash,
            dimensions=dict(nodes=nodes, edges=edges, reduction_edges=reduction_edges,
                            hidden=64, rounds=3, input_width=266),
            repeats=repeats, training_steps=steps, model_seed=17,
            index_add=_summarize(reductions), models=results,
            wall_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--nodes', type=int, default=96)
    parser.add_argument('--edges', type=int, default=2048)
    parser.add_argument('--reduction-edges', type=int, default=32768)
    parser.add_argument('--kinds', nargs='+', choices=('mlp', 'transe', 'self', 'graph'), default=['self', 'graph'])
    args = vars(parser.parse_args())
    print(json.dumps(run_probe(**args), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
