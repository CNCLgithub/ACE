"""Overlay raw gaze traces (fading trails) on pooled MOT/MIT trajectories.

Reads the unzipped OSF archive of Lukavsky & Meyerhoff (2024), exp 1:
  {data_dir}/processed/gaze.rds      raw gaze samples (xd/yd already in degrees, ttime = trajectory time)
  {data_dir}/processed/protocols.rds one row per trial: id, trial, trajectory, task (1=MOT, 2=MIT), start_time, ...
  {data_dir}/tracks.zip              tracks/T001..T400.csv (no header: col0=time s @50Hz, cols1-12 = x1,y1,...,x6,y6 in deg)

Writes, per pool trajectory, under --out-dir:
  traj{N}_mot.mp4, traj{N}_mit.mp4, traj{N}_hstack.mp4 (MOT left, MIT right)
"""
import argparse, os, re, subprocess, tempfile, zipfile, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyreadr

WARM = {1: "#e41a1c", 2: "#ff7f00", 3: "#fdb462"}   # targets
COOL = {4: "#377eb8", 5: "#984ea3", 6: "#159cc4"}   # distractors
BG = "#1a1a1a"
TRAIL_S = 1.0   # fading trail length for eye traces (seconds) — tune here
XSPAN, YSPAN = 24.0, 16.0

def render_traj(traj, prot, gaze, zf, args):
    sel_all = {}
    for task_name, task_code in (("mot", 1), ("mit", 2)):
        sel_all[task_name] = prot[(prot["trajectory"] == traj) & (prot["task"] == task_code)]
    if all(len(s) == 0 for s in sel_all.values()):
        print(f"traj {traj}: no views, skipping")
        return

    name = f"tracks/T{traj:03d}.csv"
    if name not in zf.namelist():
        print(f"traj {traj}: {name} missing from tracks.zip, skipping")
        return
    tr = pd.read_csv(io.BytesIO(zf.read(name)), header=None)
    t_traj = tr.iloc[:, 0].to_numpy(float)
    xs = [tr.iloc[:, 1 + 2 * i].to_numpy(float) for i in range(6)]
    ys = [tr.iloc[:, 2 + 2 * i].to_numpy(float) for i in range(6)]
    # no y flip: the display plugin draws yy = arenaCenterY + scale*y (y positive down),
    # and gaze yd is also positive down (yd>0 for pixel y below screen center).

    gaze_t0 = gaze[(gaze["t"] >= 0) & (gaze["t"] <= t_traj[-1])]
    prefix = os.path.join(args.out_dir, f"traj{traj:03d}")
    tmp = tempfile.mkdtemp()
    n_frames = int(t_traj[-1] * args.fps)
    for task_name in ("mot", "mit"):
        trials = sel_all[task_name][["id", "trial"]]
        gt = gaze_t0.merge(trials, on=["id", "trial"], how="inner")
        subjects = list(gt.groupby(["id", "trial"]))
        print(f"traj {traj} {task_name}: {len(gt)} samples, {len(subjects)} view-trials")
        for fi in range(n_frames):
            t_now = fi / args.fps
            fig, ax = plt.subplots(figsize=(12, 7), dpi=110)
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            ax.set_xlim(-XSPAN, XSPAN); ax.set_ylim(YSPAN, -YSPAN)
            ax.set_xticks([]); ax.set_yticks([])
            it = max(np.searchsorted(t_traj, t_now, side="right") - 1, 0)
            for oi in range(6):
                color = WARM[oi + 1] if oi < 3 else COOL[oi + 1]
                ax.add_patch(plt.Circle((xs[oi][it], ys[oi][it]), 0.35, color=color, zorder=3))
            for (sid, trl), g in subjects:
                seg = g[(g["t"] <= t_now) & (g["t"] >= t_now - TRAIL_S)]
                if len(seg) < 2: continue
                ax.plot(seg["x_dva"], seg["y_dva"], color="#dddddd", lw=1.0, alpha=0.5, zorder=1)
                ax.plot(seg["x_dva"].iloc[-1:], seg["y_dva"].iloc[-1:], "o",
                        color="#dddddd", ms=3, zorder=2)
            if fi == 0:
                ax.set_title(f"raw gaze, traj {traj}, {task_name.upper()}  (t=0)", color="w")
            fig.tight_layout()
            fig.savefig(f"{tmp}/{task_name}_f{fi:05d}.png", facecolor=BG)
            plt.close(fig)
        out = f"{prefix}_{task_name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", f"{tmp}/{task_name}_f%05d.png", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", out], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{prefix}_mot.mp4",
                    "-i", f"{prefix}_mit.mp4", "-filter_complex", "[0:v][1:v]hstack=inputs=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{prefix}_hstack.mp4"], check=True)
    print(f"traj {traj}: wrote {prefix}_hstack.mp4")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="all", help="pool trajectory index or 'all' (default)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--data-dir", default="data/mot")
    ap.add_argument("--out-dir", default="data/output/mot")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prot = list(pyreadr.read_r(f"{args.data_dir}/protocols.rds").values())[0].reset_index()
    gaze = list(pyreadr.read_r(f"{args.data_dir}/gaze.rds").values())[0]
    gaze = gaze[~gaze["out"].astype(bool)]
    gaze["t"] = gaze["ttime"].to_numpy(float)
    gaze = gaze.rename(columns={"xd": "x_dva", "yd": "y_dva"})

    zip_path = f"{args.data_dir}/tracks.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found — download it from https://osf.io/94s7t")
    with zipfile.ZipFile(zip_path) as zf:
        trajs = sorted({int(m.group(1)) for n in zf.namelist() if (m := re.search(r"T(\d+)\.csv$", n))}) \
            if args.traj == "all" else [int(args.traj)]
        for traj in trajs:
            if os.path.exists(os.path.join(args.out_dir, f"traj{traj:03d}_hstack.mp4")):
                print(f"traj {traj}: hstack already exists, skipping")
                continue
            render_traj(traj, prot, gaze, zf, args)

if __name__ == "__main__":
    main()
