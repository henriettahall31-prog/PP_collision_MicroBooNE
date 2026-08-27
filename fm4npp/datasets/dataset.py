"""
MicroBooNE Eval Dataset for FM4NPP Downstream PID Task
========================================================
Used by train/downstream/point_classification_trainer.py

Returns per-event dicts with:
  'points'     : (N, 3) [charge, wire, time] normalized, serialized by raster scan
  'reg_target' : placeholder (not used for PID task, kept for interface compatibility)
  'pid_target' : (N,) integer class label per hit, derived from G4 PDG code

PID class mapping (from g4_pdg via particle_table + edep_table g4_id link):
  0 = electron/positron (|pdg|=11)
  1 = photon (pdg=22)
  2 = muon (|pdg|=13)
  3 = proton (pdg=2212)
  4 = neutron (pdg=2112)
  5 = pion, charged or neutral (|pdg|=211 or pdg=111)
  6 = nuclear fragment (pdg >= 1e9, e.g. Argon-40 recoils)
  7 = other/unknown
"""

import numpy as np
import h5py
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Reuse the raster scan serialization and helper functions from the
# pretraining dataset file.
from fm4npp.datasets.dataset_pretrain import (
    make_quantile_bins,
    normalize,
    serialize,
    build_event_boundaries,
)

torch.manual_seed(42)

NUM_PID_CLASSES = 8


def pdg_to_class(pdg):
    """Map a G4 PDG code to one of 8 PID classes. Vectorized over a numpy array."""
    pdg_abs = np.abs(pdg)
    out = np.full(pdg.shape, 7, dtype=np.int64)  # default: other/unknown

    out[pdg_abs == 11] = 0     # electron/positron
    out[pdg == 22] = 1         # photon (no sign ambiguity)
    out[pdg_abs == 13] = 2     # muon
    out[pdg == 2212] = 3       # proton
    out[pdg == 2112] = 4       # neutron
    out[(pdg_abs == 211) | (pdg == 111)] = 5   # charged/neutral pion
    out[pdg_abs >= 1000000000] = 6             # nuclear fragment (ZZZAAA0000 format)

    return out


class MicroBooNEEvalDataset(Dataset):
    """
    MicroBooNE wire plane hit dataset with PID labels for downstream evaluation.

    For each hit, looks up its originating G4 particle via edep_table's g4_id
    (highest energy_fraction match), then maps that particle's PDG code to
    one of 8 PID classes via particle_table.
    """

    def __init__(self,
                 data_root,              # path to HDF5 file
                 plane=2,                # wire plane (0=U, 1=V, 2=Y collection)
                 train=True,
                 normalize=True,
                 limit_data=False,
                 limit_size=8000,
                 low_thr=50,
                 high_thr=10000,
                 charge_mean=175.3098,
                 charge_std=184.9520,
                 wire_min=0.0,
                 wire_max=3455.0,
                 time_min=0.0,
                 time_max=6399.0,
                 **kwargs):

        self.h5_path = data_root
        self.plane = plane
        self.train = train
        self.normalize = normalize
        self.limit_data = limit_data
        self.limit_size = limit_size
        self.low_thr = low_thr
        self.high_thr = high_thr

        self.charge_mean = charge_mean
        self.charge_std = charge_std
        self.wire_lim = {'min': wire_min, 'max': wire_max}
        self.time_lim = {'min': time_min, 'max': time_max}

        # Build event boundary index for hit_table (reused from pretrain dataset)
        self.boundaries = build_event_boundaries(self.h5_path)
        f = h5py.File(self.h5_path, 'r')
        self.total_hits = len(f['hit_table']['local_wire'])
        f.close()

        # Cache event lengths per plane (one-time cost, reused from pretraining)
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

        # Build the G4 particle_id -> pdg lookup once (small table, load fully)
        print("[INFO] Loading particle_table for PID lookup...")
        f = h5py.File(self.h5_path, 'r')
        self.particle_event_id = f['particle_table']['event_id'][:]      # (n_particles, 3)
        self.particle_g4_id = f['particle_table']['g4_id'][:].flatten()  # (n_particles,)
        self.particle_pdg = f['particle_table']['g4_pdg'][:].flatten()   # (n_particles,)
        f.close()
        print(f"[INFO] Loaded {len(self.particle_g4_id)} particle truth entries")

        self.filter_data()
        self.data_scaler = 1

    def znormalize(self, arr, mean_, std_):
        return (arr - mean_) / std_

    def minmax_normalize(self, arr, max_, min_):
        return (arr - min_) / (max_ - min_)

    def apply_norm(self, features):
        """Normalize [charge, wire, time]. Same as pretraining dataset."""
        fnorm = features.clone()
        fnorm[..., 0] = self.znormalize(fnorm[..., 0], self.charge_mean, self.charge_std)
        fnorm[..., 1] = self.minmax_normalize(fnorm[..., 1], self.wire_lim['max'], self.wire_lim['min'])
        fnorm[..., 2] = self.minmax_normalize(fnorm[..., 2], self.time_lim['max'], self.time_lim['min'])
        return fnorm

    def get_event_hits(self, event_idx):
        """Load hits for one event from hit_table, filtered to self.plane.
        Also returns the row's event_id triplet (run, subrun, event) for
        looking up matching particle_table / edep_table rows.
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
        hit_id = h['hit_id'][event_start:event_end].flatten()
        this_event_id = h['event_id'][event_start:event_start + 1][0]  # (3,) run/subrun/event
        f.close()

        mask = plane == self.plane
        wire = wire[mask].astype(np.float32)
        time = time[mask].astype(np.float32)
        charge = charge[mask].astype(np.float32)
        hit_id = hit_id[mask]

        return charge, wire, time, hit_id, this_event_id

    def get_pid_labels(self, hit_id, this_event_id):
        """
        Look up PID class for each hit via edep_table (hit_id -> g4_id,
        highest energy_fraction) then particle_table (g4_id -> g4_pdg).
        Hits with no truth match get class 7 (other/unknown).
        """
        f = h5py.File(self.h5_path, 'r')
        edep = f['edep_table']
        edep_event_id = edep['event_id'][:]        # (n_edep, 3)
        edep_hit_id = edep['hit_id'][:].flatten()
        edep_g4_id = edep['g4_id'][:].flatten()
        edep_energy_frac = edep['energy_fraction'][:].flatten()
        f.close()

        # Filter edep rows to this event
        event_mask = np.all(edep_event_id == this_event_id, axis=1)
        e_hit_id = edep_hit_id[event_mask]
        e_g4_id = edep_g4_id[event_mask]
        e_frac = edep_energy_frac[event_mask]

        # For each hit, keep the g4_id with the highest energy fraction
        hit_to_g4id = {}
        hit_to_frac = {}
        for hid, gid, frac in zip(e_hit_id, e_g4_id, e_frac):
            if hid not in hit_to_frac or frac > hit_to_frac[hid]:
                hit_to_frac[hid] = frac
                hit_to_g4id[hid] = gid

        # Build particle lookup for this event
        p_event_mask = np.all(self.particle_event_id == this_event_id, axis=1)
        p_g4id = self.particle_g4_id[p_event_mask]
        p_pdg = self.particle_pdg[p_event_mask]
        g4id_to_pdg = dict(zip(p_g4id, p_pdg))

        pdg_per_hit = np.zeros(len(hit_id), dtype=np.int64)
        for i, hid in enumerate(hit_id):
            gid = hit_to_g4id.get(hid, None)
            pdg_per_hit[i] = g4id_to_pdg.get(gid, 0) if gid is not None else 0

        pid_class = pdg_to_class(pdg_per_hit)
        return pid_class

    def filter_data(self):
        """Filter events by hit count, same as pretraining dataset."""
        self.idxlist = []
        self.seqlens = []
        tooshort, toolong = [], []
        self.longest, self.shortest = 0, 1e10

        for i in range(len(self.event_lengths)):
            n_hits = int(self.event_lengths[i])
            if n_hits < self.low_thr:
                tooshort.append(i)
            elif n_hits > self.high_thr:
                toolong.append(i)
            else:
                self.idxlist.append(i)
                self.seqlens.append(n_hits)
                self.longest = max(self.longest, n_hits)
                self.shortest = min(self.shortest, n_hits)
            if self.limit_data and len(self.idxlist) == self.limit_size:
                break

        print(f'[INFO] Filtering by N points. From {len(self.event_lengths)}, '
              f'removed short {len(tooshort)} long {len(toolong)}, remaining {len(self.idxlist)}')
        print(f'[INFO] Shortest: {self.shortest}, Longest: {self.longest}')

    def __len__(self):
        return len(self.idxlist)

    def __getitem__(self, index):
        real_idx = self.idxlist[index]

        charge, wire, time, hit_id, this_event_id = self.get_event_hits(real_idx)

        features = torch.from_numpy(
            np.stack([charge, wire, time], axis=-1)
        ).unsqueeze(0)

        if self.normalize:
            norm_features = self.apply_norm(features)
        else:
            norm_features = features

        # Serialize using the same raster scan as pretraining
        sorter = serialize(
            norm_features[0, :, 1],  # normalized wire
            norm_features[0, :, 2],  # normalized time
        )
        sorter = torch.tensor(sorter, dtype=torch.long)
        norm_features = norm_features[:, sorter].squeeze(0)
        hit_id_sorted = hit_id[sorter.numpy()]

        # Get PID labels, apply same sort order
        pid_class = self.get_pid_labels(hit_id, this_event_id)
        pid_class_sorted = torch.from_numpy(pid_class[sorter.numpy()]).long()

        # No regression target for PID task; keep zeros for interface compatibility
        reg_target = torch.zeros(norm_features.shape[0], 8)

        return {
            'points': norm_features * self.data_scaler,
            'reg_target': reg_target,
            'pid_target': pid_class_sorted,
            'mid_target': torch.zeros_like(pid_class_sorted),  # not used, kept for compatibility
        }


class MyEvalCollator(object):
    """Pads variable-length event sequences to the longest in the batch."""
    def __init__(self):
        pass

    def __call__(self, batch):
        point_longest = max(item['points'].size(0) for item in batch)
        pad_val = -100

        points, reg_targets, pid_targets, mid_targets = [], [], [], []
        for item in batch:
            n = item['points'].size(0)
            pad_n = point_longest - n
            points.append(torch.nn.functional.pad(item['points'], (0, 0, 0, pad_n), value=pad_val))
            reg_targets.append(torch.nn.functional.pad(item['reg_target'], (0, 0, 0, pad_n), value=pad_val))
            pid_targets.append(torch.nn.functional.pad(item['pid_target'], (0, pad_n), value=pad_val))
            mid_targets.append(torch.nn.functional.pad(item['mid_target'], (0, pad_n), value=pad_val))

        return {
            'points': torch.stack(points),
            'reg_target': torch.stack(reg_targets),
            'pid_target': torch.stack(pid_targets),
            'mid_target': torch.stack(mid_targets),
        }


def get_data_loader(params, distributed):
    """Create train and val data loaders for the PID downstream task."""
    train_dataset = MicroBooNEEvalDataset(
        data_root=params.data_root,
        train=True,
        normalize=True,
        limit_data=params.limit_data,
        limit_size=params.limit_size,
    )

    val_dataset = MicroBooNEEvalDataset(
        data_root=params.data_root,
        train=False,
        normalize=True,
        limit_data=True,
        limit_size=max(1, params.limit_size // 10),
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    collate_fn = MyEvalCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(params.local_batch_size),
        num_workers=getattr(params, 'num_data_workers', 2),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(params.local_valid_batch_size),
        num_workers=getattr(params, 'num_data_workers', 2),
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, train_sampler, val_loader, None
