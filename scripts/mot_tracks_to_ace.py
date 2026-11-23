#!/usr/bin/env python3
"""Convert MOT/MIT stimulus trajectories to ACE-ready CSV, one file per trajectory.

Input
  {data-dir}/tracks.zip          OSF ghd2k exp1, tracks/T001.csv ... T400.csv, no header
                                 col 0      : time in seconds, 50 Hz (20 ms steps)
                                 cols 1..12 : x1,y1,...,x6,y6 in degrees of visual angle,
                                              arena-centred, y increasing downwards
Output (--out-dir)
  ace_traj_NNN.csv.gz            one file per stimulus trajectory, long format:
                                 traj, frame, time_s, object, x, y, vx, vy, radius, is_target
  ace_tracks_meta.csv            scaling and arena metadata, one row
  ace_tracks_index.csv           per-trajectory summary: traj, n_frames, duration_s, file

Units
  Coordinates are written in ACE world units, where the graphics arena passed to
  RFGraphics((400, 400), ...) spans [-200, 200] on each axis. The scale maps the
  largest |coordinate| over all converted trajectories onto --arena-half. y is
  negated (--no-flip-y disables) because ACE treats world y as up: paint!
  renders Disc at Luxor point(pos[1], -pos[2]).

  Velocity is the forward difference pos[t+1] - pos[t] in ACE units per frame, i.e.
  the quantity WorldModel's resolve_motion adds in one step. The final frame repeats
  the previous velocity.

Dependencies: numpy, pandas.
"""
import argparse
import gzip
import io
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

N_OBJECTS = 6


def read_tracks(zip_path, trajs=None):
    """Yield (traj_index, time_s array, positions array (T, N_OBJECTS, 2))."""
    pattern = re.compile(r"T(\d+)\.csv$")
    with zipfile.ZipFile(zip_path) as zf:
        names = {}
        for name in zf.namelist():
            m = pattern.search(name)
            if m is not None:
                names[int(m.group(1))] = name
        if not names:
            raise SystemExit(f"no tracks/T###.csv entries in {zip_path}")
        wanted = sorted(names) if trajs is None else [t for t in trajs if t in names]
        missing = [] if trajs is None else [t for t in trajs if t not in names]
        for t in missing:
            print(f"warning: trajectory {t} not present in {zip_path}", file=sys.stderr)
        for traj in wanted:
            raw = np.loadtxt(io.BytesIO(zf.read(names[traj])), delimiter=",")
            t_s = raw[:, 0].astype(float)
            pos = raw[:, 1:1 + 2 * N_OBJECTS].astype(float)
            pos = pos.reshape(len(t_s), N_OBJECTS, 2)
            yield traj, t_s, pos


def global_scale(zip_path, trajs):
    peak = 0.0
    for _, _, pos in read_tracks(zip_path, trajs):
        peak = max(peak, float(np.abs(pos).max()))
    if peak <= 0.0:
        raise SystemExit("all coordinates are zero; refusing to scale")
    return peak


def long_table(traj, t_s, pos, scale, arena_half, flip_y, radius_units, n_targets):
    """Return the long-format frame x object table for one trajectory."""
    n_frames = len(t_s)

    xyz = pos * scale
    if flip_y:
        xyz[:, :, 1] = -xyz[:, :, 1]

    vel = np.empty_like(xyz)
    if n_frames > 1:
        vel[:-1] = xyz[1:] - xyz[:-1]
        vel[-1] = vel[-2]
    else:
        vel[:] = 0.0

    reach = float(np.abs(xyz).max())
    if reach > arena_half:
        print(
            f"warning: traj {traj} reaches {reach:.2f} units, beyond arena half-extent "
            f"{arena_half:.2f}; objects will clip in the receptive-field renderer",
            file=sys.stderr,
        )

    return pd.DataFrame({
        "traj": np.repeat(traj, n_frames * N_OBJECTS),
        "frame": np.tile(np.repeat(np.arange(1, n_frames + 1), N_OBJECTS), 1),
        "time_s": np.repeat(t_s, N_OBJECTS),
        "object": np.tile(np.arange(1, N_OBJECTS + 1), n_frames),
        "x": xyz[:, :, 0].reshape(-1),
        "y": xyz[:, :, 1].reshape(-1),
        "vx": vel[:, :, 0].reshape(-1),
        "vy": vel[:, :, 1].reshape(-1),
        "radius": np.full(n_frames * N_OBJECTS, radius_units),
        "is_target": np.tile(
            np.arange(1, N_OBJECTS + 1) <= n_targets, n_frames
        ).astype(int),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/mot",
                    help="directory holding tracks.zip (default: data/mot)")
    ap.add_argument("--tracks", default=None,
                    help="path to tracks.zip (overrides --data-dir)")
    ap.add_argument("--out-dir", default="data/mot/ace", help="output directory")
    ap.add_argument("--traj", default="all",
                    help="comma-separated trajectory indices, or 'all' (default)")
    ap.add_argument("--arena-half", type=float, default=200.0,
                    help="ACE world half-extent, i.e. half the RFGraphics image shape (default: 200)")
    ap.add_argument("--scale", type=float, default=None,
                    help="ACE units per degree of visual angle; default derives it from the data")
    ap.add_argument("--object-radius", type=float, default=0.35,
                    help="stimulus disc radius in degrees of visual angle (default: 0.35)")
    ap.add_argument("--n-targets", type=int, default=3,
                    help="objects 1..n are targets, used for the is_target column (default: 3)")
    ap.add_argument("--no-flip-y", action="store_true",
                    help="keep track y as given (y down) instead of negating for ACE's y-up world")
    ap.add_argument("--plain", action="store_true",
                    help="write uncompressed .csv instead of .csv.gz")
    args = ap.parse_args()

    zip_path = args.tracks or os.path.join(args.data_dir, "tracks.zip")
    if not os.path.exists(zip_path):
        raise SystemExit(f"{zip_path} not found; download it from https://osf.io/94s7t")

    trajs = None
    if args.traj != "all":
        trajs = [int(v) for v in args.traj.split(",") if v.strip()]

    peak = global_scale(zip_path, trajs)
    scale = args.scale if args.scale is not None else args.arena_half / peak
    radius_units = args.object_radius * scale
    print(f"peak |degree| = {peak:.3f}, scale = {scale:.4f} ACE units/degree, "
          f"radius = {radius_units:.3f} units")

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = ".csv" if args.plain else ".csv.gz"

    index_rows = []
    for traj, t_s, pos in read_tracks(zip_path, trajs):
        table = long_table(
            traj, t_s, pos,
            scale=scale,
            arena_half=args.arena_half,
            flip_y=not args.no_flip_y,
            radius_units=radius_units,
            n_targets=args.n_targets,
        )
        name = f"ace_traj_{traj:03d}{suffix}"
        table.to_csv(os.path.join(args.out_dir, name), index=False,
                     float_format="%.5f")
        index_rows.append({"traj": traj, "n_frames": len(t_s),
                           "duration_s": float(t_s[-1]), "file": name})

    index_path = os.path.join(args.out_dir, "ace_tracks_index.csv")
    pd.DataFrame(index_rows).to_csv(index_path, index=False)

    meta_path = os.path.join(args.out_dir, "ace_tracks_meta.csv")
    pd.DataFrame([{
        "units_per_degree": scale,
        "arena_half": args.arena_half,
        "object_radius_units": radius_units,
        "object_radius_degrees": args.object_radius,
        "y_flipped": int(not args.no_flip_y),
        "n_objects": N_OBJECTS,
        "n_targets": args.n_targets,
        "frame_hz": 50.0,
    }]).to_csv(meta_path, index=False)

    print(f"wrote {len(index_rows)} trajectory files to {args.out_dir}")
    print(f"wrote {index_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
