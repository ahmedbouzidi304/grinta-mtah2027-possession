"""Ball-possession labelling for More Than a Hack 2027, phase 1.

Rule-based, no training. Reads the tactical ATD tracking and writes a frame,label CSV.
    python src/possession.py --data data/tactical --out out/submission.csv
"""
import argparse
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
N_FRAMES = 199319
PITCH = (105.3, 68.0)
HALVES = ((11185, 90651), (110808, 190197))  # graded video-frame ranges (DATA.md)
HOME, AWAY, NONE = 0, 1, -1

PARAMS = dict(
    gap_interp=12,        # frames: ball gaps up to this are linearly interpolated
    oob_margin=0.5,       # m beyond a line before the ball counts as out
    oob_min=3,            # consecutive frames out before play is stopped
    speed_win=25,         # frames: smoothing of mean player speed
    lull_speed=1.5,       # m/s: mean player speed below this = players standing/walking
    lull_ball=3.0,        # m/s: ball slower than this (or unseen) = ball inactive
    lull_min=50,          # frames: lull must last this long to count as a stoppage
    lull_far=5.0,         # m: nobody within this distance of the ball = ball inactive
    out_side=75,          # frames: shortest stoppage after the ball crosses a touchline
    out_end=250,          # frames: shortest stoppage after the ball crosses a goal line
    wake_frames=50,       # frames of sustained player movement that prove play is on
    lull_backdate=38,     # frames: whistle precedes the slowdown by about 1.5 s
    kick_speed=7.0,       # m/s: ball faster than this has been played
    kick_follow=150,      # frames: a restart kick is followed by another kick within this
    wake_speed=1.8,       # m/s: mean player speed above this = play is on
    control_radius=1.5,   # m: ball this close to a player is under his control
    control_ball=10.0,    # m/s: a ball faster than this is not being controlled
    switch_frames=25,     # control frames the other team needs to take possession
    reset_frames=5,       # control frames by the holder that cancel a challenge
    tag_window=250,       # frames either side of a set-piece tag searched for a missed stoppage
    tag_unseen=50,        # frames the ball must be unseen/out there to call it a stoppage
    kickoff_lead=85,      # frames from a kick-off tag's start to the kick itself
    min_dead=50,          # frames: shorter stoppages are dropped
    min_alive=50,         # frames: shorter live stretches between stoppages are dropped
)


def load_teams(xml_path):
    root = ET.parse(xml_path).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == ids[0] else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def load_tracking(raw_path, n_slots):
    players = np.full((N_FRAMES, n_slots, 3), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # frame_id is 1-indexed
            players[i] = np.array([g.split(",") for g in parts[1].split(";")], dtype=np.float32)
            ball[i] = np.array(parts[2].split(",")[:2], dtype=np.float32)
    return players, ball


def load_set_pieces(xml_path):
    """[(start_frame, type)] for every SET PIECES instance in the Smart Tagging."""
    out = []
    for ins in ET.parse(xml_path).getroot().iter("instance"):
        if (ins.findtext("code") or "").strip().upper() != "SET PIECES":
            continue
        kind = ""
        for lab in ins.iter("label"):
            if (lab.findtext("group") or "").strip().lower() == "type":
                kind = (lab.findtext("text") or "").strip().lower()
        out.append((int(round(float(ins.findtext("start")) * FPS)), kind))
    return sorted(out)


def runs(mask):
    """(start, end_exclusive) of every True run."""
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def rolling_mean(x, win):
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - win), np.minimum(len(x), i + win + 1)
    cnt = n[hi] - n[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def interpolate_short_gaps(v, max_gap):
    out = v.copy()
    seen = np.isfinite(v)
    for s, e in runs(~seen):
        if s > 0 and e < len(v) and e - s <= max_gap:
            out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [v[s - 1], v[e]])
    return out


def features(players, ball, team, p):
    length, width = PITCH
    bx = interpolate_short_gaps(ball[:, 0].astype(float) * length, p["gap_interp"])
    by = interpolate_short_gaps(ball[:, 1].astype(float) * width, p["gap_interp"])
    seen = np.isfinite(bx)
    bspeed = np.full(N_FRAMES, np.nan)
    k = 2
    bspeed[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) / (2 * k / FPS)
    m = p["oob_margin"]
    oob = seen & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no tracked player
        pspeed = rolling_mean(np.nanmean(players[:, :, 2], axis=1), p["speed_win"])
    px, py = players[:, :, 0] * length, players[:, :, 1] * width
    d2 = (px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2
    d2 = np.where(np.isfinite(d2), d2, np.inf)
    nearest = d2.argmin(axis=1)
    dmin = np.sqrt(d2[np.arange(N_FRAMES), nearest])
    control = np.where((dmin <= p["control_radius"]) & ~(bspeed > p["control_ball"]),
                       team[nearest], NONE).astype(np.int8)
    control[~seen] = NONE
    end_out = seen & ((bx < -m) | (bx > length + m))
    return dict(seen=seen, bspeed=bspeed, oob=oob, end_out=end_out, pspeed=pspeed,
                control=control, dmin=dmin)


def dead_ball(f, lo, hi, p, set_pieces=()):
    """Boolean dead-ball mask for frames lo..hi (one half)."""
    n = hi - lo + 1
    seen, bspeed = f["seen"][lo:hi + 1], f["bspeed"][lo:hi + 1]
    oob, pspeed = f["oob"][lo:hi + 1], f["pspeed"][lo:hi + 1]
    end_out, dmin = f["end_out"][lo:hi + 1], f["dmin"][lo:hi + 1]

    fast = seen & (bspeed >= p["kick_speed"])
    kicks = np.array([s for s, _ in runs(fast)], dtype=int)  # frame where a played ball starts
    in_kicks = np.array([s for s in kicks if not oob[s]], dtype=int)

    # stoppage starts
    starts = []
    for s, e in runs(oob):
        if e - s >= p["oob_min"]:
            starts.append((s, p["out_end"] if end_out[s:e].any() else p["out_side"]))
    inactive = ~seen | ~(bspeed >= p["lull_ball"]) | (dmin > p["lull_far"])
    calm = (pspeed < p["lull_speed"])
    quiet = rolling_mean(inactive.astype(float), p["lull_min"] // 2) >= 0.7
    for s, e in runs(calm & quiet):
        if e - s >= p["lull_min"]:
            starts.append((max(0, s - p["lull_backdate"]), p["lull_backdate"] + p["lull_min"]))
    starts = sorted(set(starts))
    awake_runs = [(s, e) for s, e in runs(pspeed >= p["wake_speed"]) if e - s >= p["wake_frames"]]

    def restart_after(t):
        """First frame at or after t where play is back on."""
        for k in in_kicks[np.searchsorted(in_kicks, t):]:
            nxt = in_kicks[np.searchsorted(in_kicks, k + 1):]
            followed = len(nxt) and nxt[0] - k <= p["kick_follow"]
            awake = np.nanmax(pspeed[k:k + p["kick_follow"]]) >= p["wake_speed"]
            calm_again = (calm & quiet)[k + 25:k + p["kick_follow"]].mean() > 0.5 \
                if k + 25 < n else False
            if (followed or awake) and not calm_again:
                best = k
                break
        else:
            best = n
        for s, e in awake_runs:  # ball unseen at the restart: fall back on player movement
            if e - p["wake_frames"] >= t:
                best = min(best, max(t, s))
                break
        return best

    dead = np.zeros(n, bool)
    t = 0
    for s, hold in starts:
        if s < t:
            continue
        e = restart_after(s + hold)
        dead[s:e] = True
        t = e
    # Smart Tagging: a tagged set piece implies a stoppage, and a kick-off tag times its restart
    for fs, kind in set_pieces:
        t = fs - lo
        if not 0 <= t < n:
            continue
        if kind.startswith("kick off"):
            k = t + p["kickoff_lead"]
            if dead[t] and k < n:
                e = t + int(np.argmin(dead[t:])) if not dead[t:].all() else n
                if e - k > p["kickoff_lead"]:
                    cand = in_kicks[(in_kicks >= t + 50) & (in_kicks <= t + 150)]
                    dead[(cand[0] if len(cand) else k):e] = False
            continue
        a, b = max(0, t - p["tag_window"]), min(n, t + p["tag_window"])
        if dead[a:b].any():
            continue
        gone = [(e - s, s, e) for s, e in runs(~seen[a:b] | oob[a:b])]
        if gone and max(gone)[0] >= p["tag_unseen"]:
            _, s, e = max(gone)
            dead[a + s:a + e] = True
    # clean up
    for s, e in runs(~dead):
        if s > 0 and e < n and e - s < p["min_alive"]:
            dead[s:e] = True
    for s, e in runs(dead):
        if e - s < p["min_dead"]:
            dead[s:e] = False
    return dead


def possession(control, dead, p):
    """H/A/D labels for one half from control codes and the dead mask."""
    n = len(control)
    lab = np.full(n, "D", dtype="<U1")
    for s, e in runs(~dead):
        c = control[s:e]
        idx = np.flatnonzero(c != NONE)
        out = np.full(e - s, NONE, dtype=np.int8)
        if len(idx) == 0:
            lab[s:e] = "X"
            continue
        holder = c[idx[0]]
        seg_start = 0
        chal_start, chal_n, hold_n = None, 0, 0
        for i in idx:
            if c[i] == holder:
                hold_n += 1
                if hold_n >= p["reset_frames"]:
                    chal_start, chal_n = None, 0
            else:
                if chal_start is None:
                    chal_start, chal_n = i, 0
                chal_n += 1
                hold_n = 0
                if chal_n >= p["switch_frames"]:
                    out[seg_start:chal_start] = holder
                    holder, seg_start = c[i], chal_start
                    chal_start, chal_n = None, 0
        out[seg_start:] = holder
        lab[s:e] = np.where(out == HOME, "H", "A")
    return lab


def fill_x(lab):
    """Replace X by the neighbouring live label (next one first: X only opens a live stretch)."""
    for s, e in runs(lab == "X"):
        prev = lab[s - 1] if s > 0 else "D"
        nxt = lab[e] if e < len(lab) else "D"
        lab[s:e] = prev if prev in "HA" else (nxt if nxt in "HA" else "D")
    return lab


def label_match(data_dir, p=PARAMS, stem="BAR-ATH_tactical"):
    team = load_teams(os.path.join(data_dir, stem + "_atd.xml"))
    players, ball = load_tracking(os.path.join(data_dir, stem + "_atd.txt"), len(team))
    tags = load_set_pieces(os.path.join(data_dir, stem + "_smart_tagging.xml"))
    return label_arrays(players, ball, team, p, tags)


def label_arrays(players, ball, team, p=PARAMS, set_pieces=()):
    f = features(players, ball, team, p)
    labels = np.full(N_FRAMES, "D", dtype="<U1")
    for lo, hi in HALVES:
        dead = dead_ball(f, lo, hi, p, set_pieces)
        labels[lo:hi + 1] = fill_x(possession(f["control"][lo:hi + 1], dead, p))
    return labels


def write_csv(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "tactical"))
    ap.add_argument("--out", default=os.path.join("out", "submission.csv"))
    a = ap.parse_args()
    labels = label_match(a.data)
    write_csv(a.out, labels)
    print({k: int((labels == k).sum()) for k in "HAD"})
