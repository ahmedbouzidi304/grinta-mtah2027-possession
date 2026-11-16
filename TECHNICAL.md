# Technical document — Phase 1, ball possession

Team: Grinta
Code: `src/possession.py` (single file, numpy only, deterministic, about 10 s for the match)

## 1. Approach in one paragraph

The system is a rule-based pipeline with no learned parameters. It first decides, for each half,
when the ball is **dead** (stoppages), using the ball leaving the pitch and collective player
behaviour. Inside every live stretch it then tracks **which team controls the ball** and changes
possession only when the other team shows sustained control. The design target is the ranking
metric: whole possessions matter, so every rule is built to avoid splitting a possession (by a
false switch or a false stoppage) rather than to maximise per-frame accuracy.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (`_atd.txt`, `_atd.xml`) | Main input: ball position, player positions and speeds, team of each track slot |
| Smart Tagging (`_smart_tagging.xml`) | Only `SET PIECES` instances: to recover stoppages the tracking misses and to time kick-offs |
| Homography, both videos, broadcast ATD | Not used. Team identity comes from the ATD metadata, so no image work was needed |

Frame alignment: ATD `frame_id − 1` = video frame; tag seconds × 25 = video frame. Only the graded
ranges (11185–90651 and 110808–190197) are processed; every other frame is written as `D`.
The file never contains `X`.

## 3. Features (per frame)

- **Ball position** in metres (105.3 × 68.0). Gaps of up to 12 frames are linearly interpolated;
  longer gaps stay "unseen".
- **Ball speed** from positions over ±2 frames.
- **Out of play**: ball more than 0.5 m beyond a touchline or goal line.
- **Mean player speed**: mean of the speed channel over all tracked slots, smoothed over ±1 s.
  This is the main stoppage signal: players walk when play is stopped.
- **Control**: the team of the nearest tracked player, if he is within 1.5 m of the ball and the
  ball is not moving faster than 10 m/s (a fast ball passing a player is not controlled).

## 4. Possession rules

### 4.1 Dead ball

A stoppage **starts** when either:

1. the ball is out of play for at least 3 consecutive frames (start = first frame out), or
2. a *lull* lasts at least 2 s: mean player speed below 1.5 m/s while the ball is inactive
   (unseen, slower than 3 m/s, or with nobody within 5 m). The start is back-dated by 1.5 s,
   because the whistle comes before the players slow down. This is what catches fouls and
   offsides, which leave no trace in the ball track.

A stoppage **ends** at the first *restart kick*: an in-pitch ball that starts moving faster than
7 m/s and is followed, within 6 s, by another such kick or by the players speeding up above
1.8 m/s, without the lull resuming. The follow-up condition rejects the ball being rolled back
into position. Minimum stoppage lengths apply before a restart is looked for: 3 s after a
touchline exit, 10 s after a goal-line exit (goal kick or corner). If the ball is unseen at the
restart, 2 s of sustained player movement above 1.8 m/s ends the stoppage instead.

Smart Tagging adds two rules:

- A set-piece tag with no detected stoppage within ±10 s creates one on the longest unseen/out
  run of the ball in that window (if at least 2 s long). This recovers balls that leave the pitch
  untracked.
- A kick-off tag ends the stoppage about 3.4 s after the tag start (kick-off tags are 5 s windows that
  contain the kick), snapping to a detected kick when one exists.

Stoppages shorter than 2 s, and live stretches shorter than 2 s between two stoppages, are
removed.

### 4.2 Who has the ball

Each live stretch starts with the first team seen in control (the restart taker). Afterwards:

- **Loose and contested balls, passes in flight, unseen ball**: possession stays with the
  current team, as the challenge definition requires.
- **Change of possession**: the other team must accumulate 25 control frames (1 s) before the
  current team gets 5 control frames back. Control frames add up across touches, so a
  touch–pass–touch sequence counts. A single touch, a deflection or a won header without
  follow-up does not reach the threshold. The change is dated to the first frame of that
  challenge, not to when the threshold is reached, so the boundary is not late.
- **Restarts**: after every stoppage the state is reset, so `H D H` is two possessions.

## 5. Parameters and how they were chosen

All 24 constants are at the top of `src/possession.py`. They are round values chosen from
football and physics reasoning (walking pace about 1.5 m/s, a pass is faster than 7 m/s, control
means about 1 s on the ball). The public 5 minutes were used as intended: to check that frames are
aligned and that the labels mean what the ground truth means. No parameter search was run on it
and nothing was trained. A one-at-a-time sensitivity check (each parameter moved ±20–50%) keeps
the public spellF1 between 0.47 and 0.60, which is within the noise of 17 possessions.

## 6. Results

Public window (first 5 minutes of play), organisers' scorer:

| | spellF1 | macroF1 |
|---|---|---|
| baseline_smooth50 | 0.3256 | 0.6207 |
| baseline_smooth125 | 0.4242 | 0.5812 |
| this system | 0.6000 | 0.8818 |

Full match (no ground truth, structure only): 226 possessions with mean duration 14.2 s,
119 stoppages, 49% of graded frames dead, home share of live time 65%.

## 7. Limitations

- **Short possessions are missed.** A possession under about 1 s of control does not trigger a
  change, by design. The public window contains two of these.
- **Change timing.** The ground truth sometimes changes possession while the ball is still in
  flight towards the new team; this system changes at the new team's first control.
- **Stoppages with normal player movement** (a quick free kick, a quickly taken throw-in with the
  ball tracked in the pitch) are not detected.
- **Slow build-up with an unseen ball** can be mistaken for a stoppage.
- **Track slots carry the team but not the player**, so "two different players of the same team"
  cannot be used as evidence of a completed pass.
- **Team errors in the track metadata** pass straight through to the labels.
- The Smart Tagging kick-off timing (3.4 s) assumes the tag layout seen in this match.

## 8. Reproduction

    pip install -r requirements.txt
    # put the three tactical files in data/tactical/
    python src/possession.py --data data/tactical --out out/submission.csv

The output is byte-identical between runs (md5 `a169dfad2c31a02d6c2435afaaac1d37` with `\n`
line endings).
