import numpy as np
import h5py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────
# Fuction definitions
# ─────────────────────────────────────────────────────────────────────

def knn_later_indices_batch(A, k):
    """
    Find k nearest neighbors that come LATER in the sequence for each point.
    These neighbor coordinates become the prediction target for self-supervised training.
    
    A: Tensor of shape (B, N, 2) — [wire, time] coordinates per point
    k: Number of neighbors to find for each point, using only indices j > i.
    
    Returns:
        Tensor of shape (B, N, 2*k):
          - For each batch b, row i, gather up to k neighbors from rows j>i.
          - Padded with -100 if fewer than k neighbors exist.
    
    CHANGED FROM OG: D=3 → D=2, 3*k → 2*k
    """
    B, N, D = A.shape
    assert D == 2, "A must have shape (B, N, 2) for MicroBooNE [wire, time]"

    # Compute pairwise distances — shape: (B, N, N)
    A_expanded = A.unsqueeze(2)  # (B, N, 1, 2)
    A_tiled = A.unsqueeze(1)     # (B, 1, N, 2)
    pairwise_distances = torch.norm(A_expanded - A_tiled, dim=-1)  # (B, N, N)

    # Only allow neighbors with strictly larger index j>i
    mask_2d = torch.triu(torch.ones(N, N, device=A.device), diagonal=1).bool()
    mask_3d = mask_2d.unsqueeze(0).expand(B, -1, -1)
    pairwise_distances[~mask_3d] = float('inf')

    # Find k nearest among valid neighbors
    k_limited = min(k, N - 1)
    topk_vals, topk_idx = torch.topk(
        pairwise_distances,
        k=k_limited,
        dim=2,
        largest=False
    )

    # Pad if k > k_limited
    if k_limited < k:
        pad_size = k - k_limited
        inf_pad = torch.full((B, N, pad_size), float('inf'), device=A.device)
        minus1_pad = torch.full((B, N, pad_size), -1, device=A.device, dtype=torch.long)
        topk_vals = torch.cat([topk_vals, inf_pad], dim=2)
        topk_idx = torch.cat([topk_idx, minus1_pad], dim=2)

    # Mark invalid neighbors
    inf_mask = torch.isinf(topk_vals)
    topk_idx[inf_mask] = -1

    # Gather neighbor coordinates
    knn_neighbors = torch.full((B, N, k, D), -100, device=A.device, dtype=A.dtype)
    safe_idx = topk_idx.clone()
    safe_idx[safe_idx < 0] = 0
    valid_mask = (topk_idx >= 0)

    b_idx = torch.arange(B, device=A.device).view(B, 1, 1).expand(B, N, k)
    knn_neighbors[valid_mask] = A[b_idx[valid_mask], safe_idx[valid_mask], :]

    # Reshape to (B, N, 2*k)
    knn_neighbors = knn_neighbors.view(B, N, D * k)
    return knn_neighbors

# MICROBOONE HRS CODE PASTED HERE:

def make_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based binning for roughly equal hit counts per bin."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    edges = np.unique(edges)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def normalize(values: np.ndarray) -> tuple:
    """Normalize values to [0, 1]. Returns (normalized, min, max)."""
    vmin, vmax = values.min(), values.max()
    return (values - vmin) / (vmax - vmin), vmin, vmax


def serialize(wire, time, charge=None, return_details=False, n_time_bins=8, n_wire_bins=8):
    if isinstance(wire, torch.Tensor):
        wire = wire.numpy()
    if isinstance(time, torch.Tensor):
        time = time.numpy()
    N = len(wire)
    assert len(time) == N
    
    # Step 1: Normalize to [0, 1]
    time_norm, time_min, time_max = normalize(time)
    wire_norm, wire_min, wire_max = normalize(wire)
    
    # Step 2: Create bins (quantile on normalized values)
    time_edges_ = make_quantile_bins(time_norm, n_time_bins)
    wire_edges_ = make_quantile_bins(wire_norm, n_wire_bins)
    
    actual_n_time = len(time_edges_) - 1
    actual_n_wire = len(wire_edges_) - 1
    
    # Step 3: Assign each hit to a box
    time_bin = np.clip(np.digitize(time_norm, time_edges_) - 1, 0, actual_n_time - 1)
    wire_bin = np.clip(np.digitize(wire_norm, wire_edges_) - 1, 0, actual_n_wire - 1)
    
    # Composite box ID
    box_id = time_bin * actual_n_wire + wire_bin
    
    # Step 4: Compute box centers for inter-box ordering
    time_centers = 0.5 * (time_edges_[:-1] + time_edges_[1:])
    wire_centers = 0.5 * (wire_edges_[:-1] + wire_edges_[1:])
    
    unique_boxes = np.unique(box_id)
    
    box_center_time = np.zeros(len(unique_boxes))
    box_center_wire = np.zeros(len(unique_boxes))
    
    for i, bid in enumerate(unique_boxes):
        ti = bid // actual_n_wire
        wi = bid % actual_n_wire
        ti = min(ti, len(time_centers) - 1)
        wi = min(wi, len(wire_centers) - 1)
        box_center_time[i] = time_centers[ti]
        box_center_wire[i] = wire_centers[wi]
    
    # Step 5: Inter-box ordering — time → wire
    # lexsort: last key is primary, so (wire, time) means sort by time first
    inter_box_order = np.lexsort((box_center_wire, box_center_time))
    sorted_boxes = unique_boxes[inter_box_order]
    
    # Step 6: Intra-box ordering — sort by time within each box
    final_order = []
    box_assignments = []
    
    for rank, bid in enumerate(sorted_boxes):
        mask = box_id == bid
        points_in_box = np.where(mask)[0]
        
        if len(points_in_box) == 0:
            continue
        
        intra_order = np.argsort(time[points_in_box])
        sorted_points = points_in_box[intra_order]
        
        final_order.extend(sorted_points.tolist())
        box_assignments.extend([rank] * len(sorted_points))
    
    final_order = np.array(final_order, dtype=int)
    
    if return_details:
        # Convert normalized edges back to physical units for plotting
        time_edges_physical = time_edges_ * (time_max - time_min) + time_min
        wire_edges_physical = wire_edges_ * (wire_max - wire_min) + wire_min
        details = {
            'wire': wire,
            'time': time,
            'time_bin': time_bin,
            'wire_bin': wire_bin,
            'box_id': box_id,
            'box_order_rank': np.array(box_assignments),
            'time_edges': time_edges_physical,
            'wire_edges': wire_edges_physical,
            'n_occupied_boxes': len(unique_boxes),
            'n_total_boxes': actual_n_time * actual_n_wire,
        }
        return final_order, details
    
    return final_order


def group_points(arr, group_size, pad_val=-100):
    """
    Given a sequence of N x C, group them by (N//group_size+1) x group_size x C.
    Unchanged from original.
    """
    if len(arr.shape) > 2:
        arr = arr.squeeze(0)

    n, c = arr.size()
    remainder = n % group_size
    gs_ = n // group_size
    if remainder != 0:
        pad = torch.ones(group_size - remainder, c) * pad_val
        arr = torch.cat([arr, pad], dim=0)
        gs_ += 1
    return arr.reshape(gs_, group_size, c)


def build_event_boundaries(h5_path):
    """
    Scan an HDF5 file to find where each event starts.
    Saves result to a .npy cache file for fast reuse.
    Returns numpy array of boundary indices.
    """
    cache_file = h5_path.replace('.h5', '_event_boundaries.npy').replace('?download=1', '')

    if os.path.exists(cache_file):
        print(f"[INFO] Loading cached event boundaries from {cache_file}")
        return np.load(cache_file)

    print("[INFO] Building event boundary index (one-time cost)...")
    f = h5py.File(h5_path, 'r')
    total_hits = len(f['hit_table']['local_wire'])
    chunk_size = 1000000
    boundaries = [0]
    prev_last_id = None

    for start in range(0, total_hits, chunk_size):
        end = min(start + chunk_size, total_hits)
        chunk_ids = f['hit_table']['event_id'][start:end]

        if prev_last_id is not None and not np.array_equal(chunk_ids[0], prev_last_id):
            boundaries.append(start)

        changes = np.any(chunk_ids[1:] != chunk_ids[:-1], axis=1)
        change_positions = np.where(changes)[0] + 1
        for pos in change_positions:
            boundaries.append(start + pos)

        prev_last_id = chunk_ids[-1]

        if start % 10000000 == 0:
            print(f"  scanned {start}/{total_hits} rows, {len(boundaries)} events found...")

    boundaries = np.array(boundaries)
    np.save(cache_file, boundaries)
    f.close()
    print(f"[INFO] Saved {len(boundaries)} event boundaries to {cache_file}")
    return boundaries


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class MicroBooNEDataset(Dataset):
    """
    Loads 2D hits (charge, wire, time) from one wire plane of MicroBooNE
    HDF5 files and serializes them for Mamba input.
    
    Replaces the original TPCBatchDataset which loaded sPHENIX 3D data.
    """

    def __init__(self,
                 data_root,              # path to HDF5 file
                 plane=2,                # wire plane (0=U, 1=V, 2=Y collection)
                 train=True,
                 num_pred_points=10,     # k neighbors to predict (was klen)
                 normalize=True,
                 limit_data=False,
                 limit_size=8000,
                 len_chunk=512,
                 chunk_training=False,
                 # MicroBooNE normalization constants
                 charge_mean=175.3098,
                 charge_std=184.9520,
                 wire_min=0.0,
                 wire_max=3455.0,
                 time_min=0.0,
                 time_max=6399.0,
                 **kwargs):              # absorb unused sPHENIX params from config

        self.h5_path = data_root
        self.plane = plane
        self.train = train
        self.normalize = normalize
        self.num_pred_points = num_pred_points
        self.limit_data = limit_data
        self.limit_size = limit_size
        self.len_chunk = len_chunk
        self.chunk_training = chunk_training
        self.low_thr = 50

        # Normalization constants for MicroBooNE
        self.charge_mean = charge_mean
        self.charge_std = charge_std
        self.wire_lim = {'min': wire_min, 'max': wire_max}
        self.time_lim = {'min': time_min, 'max': time_max}

        # Build event boundary index
        self.boundaries = build_event_boundaries(self.h5_path)
        # Get total hit count for computing last event's length
        f = h5py.File(self.h5_path, 'r')
        self.total_hits = len(f['hit_table']['local_wire'])
        f.close()

        # Cache event lengths per plane (one-time cost)
        length_cache = self.h5_path.replace('.h5', f'_plane{self.plane}_lengths.npy').replace('?download=1', '')
        if os.path.exists(length_cache):
            print(f"[INFO] Loading cached event lengths from {length_cache}")
            self.event_lengths = np.load(length_cache)
        else:
            print("[INFO] Computing event lengths per plane (one-time cost)...")
            f = h5py.File(self.h5_path, 'r')
            plane_col = f['hit_table']['local_plane']
            lengths = []
            for i in range(len(self.boundaries)):
                start = int(self.boundaries[i])
                end = int(self.boundaries[i + 1]) if i + 1 < len(self.boundaries) else self.total_hits
                p = plane_col[start:end].flatten()
                lengths.append(int((p == self.plane).sum()))
                if i % 5000 == 0:
                    print(f"  computed {i}/{len(self.boundaries)} event lengths...")
            f.close()
            self.event_lengths = np.array(lengths)
            np.save(length_cache, self.event_lengths)
            print(f"[INFO] Saved {len(self.event_lengths)} event lengths to {length_cache}")

        # Filter events by hit count
        self.filter_data()

        self.data_scaler = 1

    def znormalize(self, arr, mean_, std_):
        return (arr - mean_) / std_

    def minmax_normalize(self, arr, max_, min_):
        return (arr - min_) / (max_ - min_)

    def apply_norm(self, features):
        """
        Normalize [charge, wire, time] features.
        Charge: z-normalize (subtract mean, divide by std)
        Wire: min-max normalize to [0, 1]
        Time: min-max normalize to [0, 1]
        
        Changed from original: was [E, eta, phi, r] (4 features) → [charge, wire, time] (3 features)
        """
        fnorm = features.clone()
        fnorm[..., 0] = self.znormalize(fnorm[..., 0], self.charge_mean, self.charge_std)
        fnorm[..., 1] = self.minmax_normalize(fnorm[..., 1], self.wire_lim['max'], self.wire_lim['min'])
        fnorm[..., 2] = self.minmax_normalize(fnorm[..., 2], self.time_lim['max'], self.time_lim['min'])
        return fnorm

    def get_event_hits(self, event_idx):
        """
        Load hits for one event from HDF5, filtered to self.plane.
        Returns (charge, wire, time) as float32 numpy arrays.
        
        Replaces the original's memmap loading.
        """
        event_start = int(self.boundaries[event_idx])
        if event_idx + 1 < len(self.boundaries):
            event_end = int(self.boundaries[event_idx + 1])
        else:
            event_end = self.total_hits

        f = h5py.File(self.h5_path, 'r')
        h = f['hit_table']
        plane = h['local_plane'][event_start:event_end].flatten()
        wire = h['local_wire'][event_start:event_end].flatten()
        time = h['local_time'][event_start:event_end].flatten()
        charge = h['integral'][event_start:event_end].flatten()
        f.close()

        # Filter to selected plane
        mask = plane == self.plane
        wire = wire[mask].astype(np.float32)
        time = time[mask].astype(np.float32)
        charge = charge[mask].astype(np.float32)

        return charge, wire, time

    def filter_data(self, low_thr=-1, high_thr=10000):
        """
        Filter out events with too few or too many hits.
        Uses cached self.event_lengths instead of reading HDF5 per event.
        """
        self.idxlist = []
        self.seqlens = []
        self.tooshort = []
        self.toolong = []
        self.longest = 0
        self.shortest = 1e10

        for i in range(len(self.event_lengths)):
            n_hits = int(self.event_lengths[i])

            if n_hits < low_thr:
                self.tooshort.append(i)
            elif n_hits > high_thr:
                self.toolong.append(i)
            else:
                self.idxlist.append(i)
                self.seqlens.append(n_hits)

                if self.longest < n_hits:
                    self.longest = n_hits
                if self.shortest > n_hits:
                    self.shortest = n_hits

            if self.limit_data and len(self.idxlist) == self.limit_size:
                break

        print(f'[INFO] Filtering by N points. From {len(self.event_lengths)}, '
              f'removed short {len(self.tooshort)} long {len(self.toolong)}, '
              f'remaining {len(self.idxlist)}')
        print(f'[INFO] Shortest: {self.shortest}, Longest: {self.longest}')

    def cut_chunk(self, sequence, maxlen):
        """
        Apply chunk-based training.
        If seq_len > maxlen, cut a sub-chunk from a random location.
        Unchanged from original.
        """
        N, D = sequence.shape
        start_idx = 0

        if maxlen > N:
            return sequence, start_idx
        else:
            start_idx = torch.randint(0, N - self.low_thr + 1, (1,)).item()
            chunk = sequence[start_idx: start_idx + maxlen]
            return chunk, start_idx

    def __len__(self):
        return len(self.idxlist)

    def __getitem__(self, index):
        """
        Load one event, normalize, serialize by raster scan, find kNN prediction targets.
        
        Key changes from original:
          - No cartesian_to_polar conversion (removed)
          - 3 features [charge, wire, time] instead of 4 [E, eta, phi, r]
          - 2D raster scan serialization instead of 3D voxelizer
          - 2D kNN instead of 3D
        """
        real_idx = self.idxlist[index]

        # Load one event's hits from HDF5
        charge, wire, time = self.get_event_hits(real_idx)

        # Stack into (1, N, 3) = [charge, wire, time]
        features = torch.from_numpy(
            np.stack([charge, wire, time], axis=-1)
        ).unsqueeze(0)

        # No truth labels for pretraining — placeholder
        target = torch.zeros(1, features.shape[1], dtype=torch.long)

        # Normalize
        if self.normalize:
            norm_features = self.apply_norm(features)
        else:
            norm_features = features

        # Serialize using 2D hierarchical raster scan on normalized coords
        sorter = serialize(
            norm_features[0, :, 1],  # normalized wire
            norm_features[0, :, 2],  # normalized time
        )
        sorter = torch.tensor(sorter, dtype=torch.long)
        norm_features = norm_features[:, sorter]
        norm_target = target[:, sorter]

        # Find k nearest later neighbors for self-supervised prediction target
        # Uses only positional dims (wire, time), not charge
        knearest_points = knn_later_indices_batch(
            norm_features[..., 1:3], k=self.num_pred_points
        )

        # Squeeze batch dim for collator
        serialized_points = norm_features.squeeze(0)
        knearest_points = knearest_points.squeeze(0)
        serialized_target = norm_target.squeeze(0)

        return serialized_points * self.data_scaler, serialized_target, knearest_points * self.data_scaler


# ─────────────────────────────────────────────────────────────────────
# Collator (pads variable-length sequences to same length in a batch)
# ─────────────────────────────────────────────────────────────────────

class MyCollator(object):
    """
    Unchanged from original.
    Pads variable-length event sequences to the longest in the batch.
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        # Find longest sequence in this batch
        point_longest = 0
        for g, t, k in batch:
            if point_longest < g.size(0):
                point_longest = g.size(0)

        grouped, targets, knearest = [], [], []

        pad_val = -100
        for g, t, k in batch:
            grouped.append(torch.nn.functional.pad(g, (0, 0, 0, point_longest - g.size(0)), value=pad_val))
            targets.append(torch.nn.functional.pad(t, (0, point_longest - g.size(0)), value=pad_val))
            knearest.append(torch.nn.functional.pad(k, (0, 0, 0, point_longest - g.size(0)), value=pad_val))

        grouped = torch.stack(grouped)
        targets = torch.stack(targets)
        knearest = torch.stack(knearest)

        return (grouped, targets, knearest)


# ─────────────────────────────────────────────────────────────────────
# Data loader factory
# ─────────────────────────────────────────────────────────────────────

def get_data_loader(params, distributed):
    """
    Create train and test data loaders.
    
    Changed from original: uses MicroBooNEDataset instead of TPCBatchDataset.
    Removed sPHENIX-specific params (order, voxelize, bin_dir, etc.)
    """
    train_dataset = MicroBooNEDataset(
        data_root=params.data_root,
        train=True,
        num_pred_points=params.klen,
        normalize=True,
        limit_data=params.limit_data,
        limit_size=params.limit_size,
        len_chunk=params.len_chunk,
        chunk_training=params.chunk_training,
    )

    test_dataset = MicroBooNEDataset(
        data_root=params.data_root,
        train=False,
        num_pred_points=params.klen,
        normalize=True,
        len_chunk=params.len_chunk,
        chunk_training=params.chunk_training,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    my_collate_fn = MyCollator()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(params.local_batch_size),
        num_workers=params.num_data_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=my_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=int(params.local_valid_batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=my_collate_fn,
    )

    return train_dataloader, train_sampler, test_dataloader, test_sampler
