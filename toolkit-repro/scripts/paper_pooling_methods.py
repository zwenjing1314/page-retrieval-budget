"""Paper-derived ColPali pooling adapters; not original paper experiment code.

Cosine Lloyd loop follows the initialization/update/stopping operations of
subhadarship/kmeans_pytorch, commit f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805.
Its MIT license is retained in the run's sources/kmeans-pytorch-pinned-LICENSE.txt.
That package is a third-party reference, not an identified implementation used
by the Token Pooling paper. No package installation or model access is required.
"""
import numpy as np
import torch


class PoolingFailure(RuntimeError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def aggregate(full, labels, k):
    x = np.asarray(full, dtype=np.float32)
    sizes = np.bincount(labels, minlength=k)
    assert len(labels) == 1024 and len(sizes) == k and (sizes > 0).all()
    raw = np.stack([x[labels == i].mean(axis=0, dtype=np.float32) for i in range(k)])
    norms = np.linalg.norm(raw, axis=1)
    assert np.isfinite(raw).all() and np.all(norms > 0)
    pooled = (raw / norms[:, None]).astype(np.float16)
    assert pooled.shape == (34, 128) and np.isfinite(pooled).all()
    stored_norms = np.linalg.norm(pooled.astype(np.float32), axis=1)
    assert np.max(np.abs(stored_norms - 1)) <= 0.005
    return dict(pooled=pooled, raw_means=raw, labels=labels.astype(np.int32),
                group_sizes=sizes.astype(np.int32)), dict(
        raw_mean_norm_min=float(norms.min()), raw_mean_norm_max=float(norms.max()),
        stored_norm_min=float(stored_norms.min()), stored_norm_max=float(stored_norms.max()),
        max_stored_unit_norm_error=float(np.max(np.abs(stored_norms - 1))))


def sequential(full):
    assert full.shape == (1024, 128) and full.dtype == np.float16
    labels = np.arange(1024, dtype=np.int32) // 31
    arrays, stats = aggregate(full, labels, 34)
    assert arrays["group_sizes"].tolist() == [31] * 33 + [1]
    return arrays, dict(**stats, group_width=31, iterations=0, converged=True)


def cosine_kmeans(full, seed, tol=1e-4, max_iterations=1000):
    assert full.shape == (1024, 128) and full.dtype == np.float16
    x = torch.from_numpy(full.astype(np.float32))
    # RandomState is the same MT19937 selection as np.random.seed/choice in the
    # pinned reference, without altering the caller's global random generator.
    initial_indices = np.random.RandomState(seed).choice(1024, 34, replace=False)
    centers = x[initial_indices].clone()
    trace = []
    for iteration in range(1, max_iterations + 1):
        # Preserve the pinned reference's cosine broadcasting and reductions.
        left, right = x.unsqueeze(1), centers.unsqueeze(0)
        distances = 1 - ((left / left.norm(dim=-1, keepdim=True)) *
                         (right / right.norm(dim=-1, keepdim=True))).sum(dim=-1)
        assert torch.isfinite(distances).all()
        labels = distances.argmin(dim=1)
        sizes = torch.bincount(labels, minlength=34)
        empty = torch.where(sizes == 0)[0].tolist()
        if empty:
            raise PoolingFailure("Empty cluster; no reseeding or cluster removal", dict(
                seed=seed, iteration=iteration, empty_cluster_ids=empty,
                initial_indices=initial_indices.tolist(), trace=trace, sizes=sizes.tolist()))
        previous = centers.clone()
        for i in range(34):
            centers[i] = x[labels == i].mean(dim=0)
        shift = torch.sqrt(((centers - previous) ** 2).sum(dim=1)).sum()
        squared_shift = float(shift ** 2)
        trace.append(dict(iteration=iteration, squared_center_shift=squared_shift,
                          smallest_cluster=int(sizes.min()), largest_cluster=int(sizes.max())))
        if squared_shift < tol:
            labels_np = labels.numpy()
            arrays, stats = aggregate(full, labels_np, 34)
            # Means are recomputed from members, as required by the paper.
            assert np.max(np.abs(arrays["raw_means"] - centers.numpy())) <= 1e-6
            return arrays, dict(**stats, seed=seed, initial_indices=initial_indices.tolist(),
                                iterations=iteration, stopping_value=squared_shift,
                                stopping_threshold=tol, converged=True, empty_clusters=0, trace=trace)
    raise PoolingFailure("Iteration guard reached without reference convergence", dict(
        seed=seed, iterations=max_iterations, trace=trace, initial_indices=initial_indices.tolist()))
