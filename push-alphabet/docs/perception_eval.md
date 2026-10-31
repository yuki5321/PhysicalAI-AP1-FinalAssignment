# Perception evaluation: letter-detection recall (before / after)

Measurement log for the miss-rate work (segmentation contract: [cnn.md](cnn.md) "前段の分割").
Everything here is reproducible with the commands below; `data/` is not in git
(`tools/gen_dataset.py --source synth --out data/synth_val --frames 1000 --seed 2`).

## Tools

| tool | what |
|---|---|
| `tools/eval_perception.py` | accuracy / recall / missed / false / pose error / ms per method; subsets `all` and `occ<=0.1` ("visible"). New: `--per-letter` (I/U/Y/K breakdown), `--json [PATH]` (machine-readable summary incl. `precision`, `per_letter`, `confusion`), `--src ROOT` (import pa_core / pa_perception from another checkout, e.g. a baseline worktree). CSV gained a trailing `precision` column; other output unchanged. |
| `tools/analyze_misses.py` | classifies every recall loss and false positive by cause (see its docstring); `--out DIR` writes `<method>_misses.png` contact sheets, `<method>_records.csv`, `summary.json`, `tables.md` (keep DIR outside git). |
| `ros2_ws/src/pa_perception/test/test_recall_regression.py` | pytest, ~17 s: 120 synth frames rendered on the fly (gen_dataset, seed 7), recall / precision floors per method (cnn / hybrid skipped without onnxruntime / model). Floors = latest results minus a margin; raise `FLOORS` when results improve. |

```bash
export PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception
python3 tools/eval_perception.py --data data/synth_val --methods classic,cnn,hybrid \
    --per-letter --markdown --csv out/eval.csv --json out/eval.json
python3 tools/analyze_misses.py --data data/synth_val --methods classic,cnn,hybrid --out /tmp/misses
# baseline from a pristine checkout:
git worktree add /tmp/base c1664f5
python3 tools/eval_perception.py --src /tmp/base --data data/synth_val ...
```

Definitions: `recall = correct / labels` (wrong class and unknown count as losses),
`precision = correct / (matched + false)`, `missed` = labels with no detection within
30 mm, `false` = unmatched detections. Labels outside the workspace (+20 mm) are not
counted. `occ` = fraction of the top face hidden by a paddle / hand.

## Baseline (main @ c1664f5)

data/synth_val, **full 1000 frames** (1659 labelled letters, 1644 counted; 545 with occ > 0.1),
model `ros2_ws/src/pa_perception/models/letter_cnn.onnx` as of c1664f5. 4-core container.

| method | subset | frames | labels | acc | recall | precision | missed | false | pos_mean | pos_p95 | pos_max | ang_mean | ang_p95 | ang_max | ms/frame |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| classic | all | 1000 | 1644 | 0.9313 | 0.6928 | 0.9127 | 421 | 25 | 1.29 | 6.66 | 15.83 | 0.47 | 2.83 | 8.60 | 21.7 |
| classic | occ<=0.1 | 1000 | 1110 | 0.9942 | 0.9189 | 0.9705 | 84 | 25 | 0.57 | 3.39 | 8.28 | 0.32 | 1.88 | 8.60 | 21.7 |
| cnn | all | 1000 | 1644 | 0.9574 | 0.7251 | 0.9423 | 399 | 20 | 0.53 | 1.95 | 10.89 | 1.60 | 4.66 | 11.28 | 8.2 |
| cnn | occ<=0.1 | 1000 | 1110 | 0.9971 | 0.9261 | 0.9781 | 79 | 20 | 0.30 | 0.84 | 3.09 | 1.38 | 3.96 | 10.89 | 8.2 |
| hybrid | all | 1000 | 1644 | 0.9564 | 0.7476 | 0.9367 | 359 | 27 | 0.61 | 2.35 | 10.89 | 0.31 | 1.49 | 10.21 | 42.8 |
| hybrid | occ<=0.1 | 1000 | 1110 | 0.9971 | 0.9315 | 0.9718 | 73 | 27 | 0.30 | 0.88 | 3.09 | 0.18 | 0.57 | 10.21 | 42.8 |

(pos in mm, angle in deg, symmetry-aware.) Confusion (true -> predicted, all labels):
classic U->I 48, Y->I 19, K->Y 14, K->I 3; cnn U->I 53; hybrid U->I 56. Every false
positive of every method is reported as `I` (a straight fragment: a paddle-cut arm of
another letter, or a paddle / hand edge).

### Baseline per letter

`wrong` = matched with another letter (or unknown); `false` = false positives by predicted letter.

| method | subset | letter | labels | recall | missed | wrong | false |
|---|---|---|--:|--:|--:|--:|--:|
| classic | all | I | 415 | 0.8361 | 68 | 0 | 25 |
| classic | all | U | 412 | 0.5680 | 130 | 48 | 0 |
| classic | all | Y | 409 | 0.6797 | 112 | 19 | 0 |
| classic | all | K | 408 | 0.6863 | 111 | 17 | 0 |
| classic | occ<=0.1 | I | 292 | 0.9658 | 10 | 0 | 25 |
| classic | occ<=0.1 | U | 252 | 0.8690 | 31 | 2 | 0 |
| classic | occ<=0.1 | Y | 285 | 0.9193 | 22 | 1 | 0 |
| classic | occ<=0.1 | K | 281 | 0.9146 | 21 | 3 | 0 |
| cnn | all | I | 415 | 0.8193 | 75 | 0 | 20 |
| cnn | all | U | 412 | 0.6117 | 107 | 53 | 0 |
| cnn | all | Y | 409 | 0.7335 | 109 | 0 | 0 |
| cnn | all | K | 408 | 0.7353 | 108 | 0 | 0 |
| cnn | occ<=0.1 | I | 292 | 0.9658 | 10 | 0 | 20 |
| cnn | occ<=0.1 | U | 252 | 0.8889 | 25 | 3 | 0 |
| cnn | occ<=0.1 | Y | 285 | 0.9123 | 25 | 0 | 0 |
| cnn | occ<=0.1 | K | 281 | 0.9324 | 19 | 0 | 0 |
| hybrid | all | I | 415 | 0.8361 | 68 | 0 | 27 |
| hybrid | all | U | 412 | 0.6262 | 98 | 56 | 0 |
| hybrid | all | Y | 409 | 0.7628 | 97 | 0 | 0 |
| hybrid | all | K | 408 | 0.7647 | 96 | 0 | 0 |
| hybrid | occ<=0.1 | I | 292 | 0.9658 | 10 | 0 | 27 |
| hybrid | occ<=0.1 | U | 252 | 0.9008 | 22 | 3 | 0 |
| hybrid | occ<=0.1 | Y | 285 | 0.9158 | 24 | 0 | 0 |
| hybrid | occ<=0.1 | K | 281 | 0.9395 | 17 | 0 | 0 |

### Regression-test baseline (seed 7, 120 frames, on the fly)

| method | visible recall | visible precision | all recall | floors (vis recall / vis precision / all recall) |
|---|--:|--:|--:|---|
| classic | 0.9552 | 0.9846 | 0.6698 | 0.92 / 0.95 / 0.64 (raised after the fix, see below) |
| cnn | 0.9478 | 0.9845 | 0.6840 | 0.92 / 0.95 / 0.65 |
| hybrid | 0.9552 | 0.9771 | 0.7217 | 0.92 / 0.95 / 0.69 |

### Baseline miss causes (tools/analyze_misses.py, full 1000 frames)

Causes recomputed from `ClassicDetector.segment` + connected components (the
segmentation both cnn and hybrid use at c1664f5), the ground-truth silhouettes and an
HSV occluder mask (warm hue, S >= 60, V >= 70). "total lost" = labels x (1 - recall)
(missed + wrong class). Category rules: see the docstring of tools/analyze_misses.py.

#### Recall losses by cause (counted labels; all / occ<=0.1)

| cause | classic all | cnn all | hybrid all | classic vis | cnn vis | hybrid vis |
|---|--:|--:|--:|--:|--:|--:|
| labels | 1644 | 1644 | 1644 | 1110 | 1110 | 1110 |
| recall | 0.6928 | 0.7251 | 0.7476 | 0.9189 | 0.9261 | 0.9315 |
| wrong_class | 84 | 53 | 56 | 6 | 3 | 3 |
| merged_letter | 28 | 28 | 28 | 12 | 12 | 12 |
| merged_occluder | 179 | 176 | 175 | 57 | 56 | 55 |
| mislocalized | 0 | 0 | 0 | 0 | 0 | 0 |
| heavy_occ | 39 | 39 | 38 | 0 | 0 | 0 |
| no_foreground | 0 | 0 | 0 | 0 | 0 | 0 |
| fragmented | 69 | 78 | 71 | 4 | 3 | 3 |
| area_small | 1 | 1 | 1 | 0 | 0 | 0 |
| area_large | 0 | 0 | 0 | 0 | 0 | 0 |
| outside_ws | 0 | 0 | 0 | 0 | 0 | 0 |
| shape_rejected | 105 | 77 | 46 | 11 | 8 | 3 |
| **total lost** | 505 | 452 | 415 | 90 | 82 | 76 |

#### Mechanical cause by occlusion bucket (all labels; heavy_occ not applied)

| cause | classic occ<=0.1 | classic 0.1<occ<=0.3 | classic occ>0.3 | cnn occ<=0.1 | cnn 0.1<occ<=0.3 | cnn occ>0.3 | hybrid occ<=0.1 | hybrid 0.1<occ<=0.3 | hybrid occ>0.3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| wrong_class | 6 | 76 | 2 | 3 | 48 | 2 | 3 | 51 | 2 |
| merged_letter | 12 | 14 | 2 | 12 | 14 | 2 | 12 | 14 | 2 |
| merged_occluder | 57 | 105 | 17 | 56 | 104 | 16 | 55 | 104 | 16 |
| mislocalized | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| no_foreground | 0 | 0 | 14 | 0 | 0 | 14 | 0 | 0 | 14 |
| fragmented | 4 | 65 | 15 | 3 | 75 | 15 | 3 | 68 | 15 |
| area_small | 0 | 1 | 5 | 0 | 1 | 5 | 0 | 1 | 5 |
| area_large | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| outside_ws | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| shape_rejected | 11 | 94 | 5 | 8 | 69 | 5 | 3 | 43 | 4 |

#### False positives by cause (all labels)

| cause | classic | cnn | hybrid |
|---|--:|--:|--:|
| false (eval) | 25 | 20 | 27 |
| duplicate | 12 | 11 | 14 |
| mislocalized | 8 | 0 | 0 |
| occluder | 5 | 9 | 13 |
| background | 0 | 0 | 0 |

Reading of the baseline:

- **merged_occluder is the largest single cause** (175-179 of 415-505 losses, and 55-57 of
  the 76-90 losses even on *visible* letters): the paddle (bright orange passes the gray
  threshold) or its side wall touches the letter and the blob is merged, so area / shape
  checks fail. Colour-aware foreground (exclude occluder pixels) should recover most of these.
- **fragmented** (69-78) and **shape_rejected** (46-105) are almost all at 0.1 < occ <= 0.3:
  a paddle lying across a letter cuts it into pieces or removes part of the silhouette;
  needs occlusion-tolerant matching (occluder pixels = unknown in the IoU). hybrid already
  recovers ~60 of classic's shape rejects.
- **wrong_class** is U -> I (48-56): a paddle-cut U whose biggest remaining piece is one arm.
- **merged_letter** (28, 12 visible): touching letters (generator allows 0 px gap); needs
  blob splitting.
- **heavy_occ** (occ > 0.3, 38-39) and no_foreground (14, all occ > 0.3) are expected losses.
- False positives are all reported as `I`: **duplicate** (11-14) = a second detection on a
  letter already matched (a paddle-cut fragment), **occluder** (5-13) = a whole paddle arm
  read as an I; classic's 8 **mislocalized** are wrong poses within 60 mm of an unmatched label.

Example contact sheets (not in git): `analyze_misses.py ... --out DIR` -> `DIR/<method>_misses.png`
(green = true top face, yellow = foreground contours, red tint = occluder mask, magenta = detection centroids).

## After (HEAD afaeea7: segmentation v2 + occlusion-tolerant matching)

Same data/synth_val, full 1000 frames, same model file, working tree = HEAD afaeea7
(d5fecf2 "color-aware segmentation with occluder mask, background model and blob
split/merge", dd71000 "occlusion-tolerant matching, v2 segmentation", afaeea7). Measured after
the detector sources had been unchanged for 10 min and data/READY_segmentation existed.

| method | subset | labels | acc | recall | precision | missed | false | pos_mean | pos_p95 | pos_max | ang_mean | ang_p95 | ang_max | ms/frame |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| classic | all | 1644 | 0.9994 | 0.9635 | 0.9987 | 59 | 1 | 0.42 | 2.27 | 5.96 | 0.30 | 1.52 | 4.82 | 28.4 |
| classic | occ<=0.1 | 1110 | 1.0000 | 0.9973 | 0.9991 | 3 | 1 | 0.20 | 0.61 | 4.88 | 0.17 | 0.90 | 2.68 | 28.4 |
| cnn | all | 1644 | 0.9981 | 0.9678 | 0.9981 | 50 | 0 | 0.63 | 2.62 | 10.97 | 1.85 | 6.14 | 16.79 | 18.8 |
| cnn | occ<=0.1 | 1110 | 1.0000 | 0.9973 | 1.0000 | 3 | 0 | 0.31 | 0.90 | 3.22 | 1.40 | 4.16 | 11.66 | 18.8 |
| hybrid | all | 1644 | 0.9994 | 0.9659 | 0.9994 | 55 | 0 | 0.67 | 2.76 | 10.97 | 0.43 | 2.12 | 16.10 | 29.9 |
| hybrid | occ<=0.1 | 1110 | 1.0000 | 0.9964 | 1.0000 | 4 | 0 | 0.30 | 0.90 | 3.22 | 0.14 | 0.46 | 10.27 | 29.9 |

### Before / after

ms/frame is noisy: both runs shared a 4-core container with other agents' jobs.

| method | subset | recall before | recall after | Δ | precision before | precision after | missed before → after | false before → after | pos_p95 mm | ang_p95 ° | ms/frame |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| classic | all | 0.6928 | 0.9635 | +0.2707 | 0.9127 | 0.9987 | 421 → 59 | 25 → 1 | 6.66 → 2.27 | 2.83 → 1.52 | 21.7 → 28.4 |
| classic | occ<=0.1 | 0.9189 | 0.9973 | +0.0784 | 0.9705 | 0.9991 | 84 → 3 | 25 → 1 | 3.39 → 0.61 | 1.88 → 0.90 | 21.7 → 28.4 |
| cnn | all | 0.7251 | 0.9678 | +0.2427 | 0.9423 | 0.9981 | 399 → 50 | 20 → 0 | 1.95 → 2.62 | 4.66 → 6.14 | 8.2 → 18.8 |
| cnn | occ<=0.1 | 0.9261 | 0.9973 | +0.0712 | 0.9781 | 1.0000 | 79 → 3 | 20 → 0 | 0.84 → 0.90 | 3.96 → 4.16 | 8.2 → 18.8 |
| hybrid | all | 0.7476 | 0.9659 | +0.2184 | 0.9367 | 0.9994 | 359 → 55 | 27 → 0 | 2.35 → 2.76 | 1.49 → 2.12 | 42.8 → 29.9 |
| hybrid | occ<=0.1 | 0.9315 | 0.9964 | +0.0649 | 0.9718 | 1.0000 | 73 → 4 | 27 → 0 | 0.88 → 0.90 | 0.57 → 0.46 | 42.8 → 29.9 |

### Per-letter recall, before / after

| method | letter | all: before → after | occ<=0.1: before → after |
|---|---|--:|--:|
| classic | I | 0.8361 → 0.9157 | 0.9658 → 1.0000 |
| classic | U | 0.5680 → 0.9733 | 0.8690 → 0.9960 |
| classic | Y | 0.6797 → 0.9804 | 0.9193 → 0.9930 |
| classic | K | 0.6863 → 0.9853 | 0.9146 → 1.0000 |
| cnn | I | 0.8193 → 0.9373 | 0.9658 → 1.0000 |
| cnn | U | 0.6117 → 0.9709 | 0.8889 → 0.9960 |
| cnn | Y | 0.7335 → 0.9780 | 0.9123 → 0.9930 |
| cnn | K | 0.7353 → 0.9853 | 0.9324 → 1.0000 |
| hybrid | I | 0.8361 → 0.9277 | 0.9658 → 1.0000 |
| hybrid | U | 0.6262 → 0.9733 | 0.9008 → 0.9960 |
| hybrid | Y | 0.7628 → 0.9756 | 0.9158 → 0.9895 |
| hybrid | K | 0.7647 → 0.9877 | 0.9395 → 1.0000 |

### Miss causes after (tools/analyze_misses.py)

Causes now come from `segmentation.segment_letters` (`--seg auto`), i.e. what the
detectors see; occluder pixels are no longer foreground, so merged_occluder and fragmented
went to 0 by construction **and** by recall (the recall row is from the detectors).

#### Recall losses by cause (counted labels; all / occ<=0.1)

| cause | classic all | cnn all | hybrid all | classic vis | cnn vis | hybrid vis |
|---|--:|--:|--:|--:|--:|--:|
| labels | 1644 | 1644 | 1644 | 1110 | 1110 | 1110 |
| recall | 0.9635 | 0.9678 | 0.9659 | 0.9973 | 0.9973 | 0.9964 |
| wrong_class | 1 | 3 | 1 | 0 | 0 | 0 |
| merged_letter | 0 | 0 | 0 | 0 | 0 | 0 |
| merged_occluder | 0 | 0 | 0 | 0 | 0 | 0 |
| mislocalized | 0 | 0 | 0 | 0 | 0 | 0 |
| heavy_occ | 32 | 31 | 31 | 0 | 0 | 0 |
| no_foreground | 0 | 0 | 0 | 0 | 0 | 0 |
| fragmented | 0 | 0 | 0 | 0 | 0 | 0 |
| area_small | 1 | 1 | 1 | 0 | 0 | 0 |
| area_large | 0 | 0 | 0 | 0 | 0 | 0 |
| outside_ws | 0 | 0 | 0 | 0 | 0 | 0 |
| shape_rejected | 26 | 18 | 23 | 3 | 3 | 4 |
| **total lost** | 60 | 53 | 56 | 3 | 3 | 4 |

#### Mechanical cause by occlusion bucket (all labels; heavy_occ not applied)

| cause | classic occ<=0.1 | classic 0.1<occ<=0.3 | classic occ>0.3 | cnn occ<=0.1 | cnn 0.1<occ<=0.3 | cnn occ>0.3 | hybrid occ<=0.1 | hybrid 0.1<occ<=0.3 | hybrid occ>0.3 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| wrong_class | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 1 |
| merged_letter | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| merged_occluder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mislocalized | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| no_foreground | 0 | 0 | 15 | 0 | 0 | 15 | 0 | 0 | 15 |
| fragmented | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| area_small | 0 | 1 | 14 | 0 | 1 | 13 | 0 | 1 | 13 |
| area_large | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| outside_ws | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| shape_rejected | 3 | 23 | 3 | 3 | 15 | 3 | 4 | 19 | 3 |

#### False positives by cause (all labels)

| cause | classic | cnn | hybrid |
|---|--:|--:|--:|
| false (eval) | 1 | 0 | 0 |
| duplicate | 0 | 0 | 0 |
| mislocalized | 0 | 0 | 0 |
| occluder | 1 | 0 | 0 |
| background | 0 | 0 | 0 |

Reading of the after numbers:

- Recall loss fell from 415-505 to 53-60 labels (all) and from 76-90 to 3-4 (visible).
  merged_occluder (175-179), fragmented (69-78), merged_letter (28) and duplicate /
  occluder false positives are gone; U -> I confusions fell from 48-56 to 1-3.
- What is left: **heavy_occ** (31-32, occ > 0.3, mostly area_small / no_foreground: less than
  ~60 % of the letter visible) and **shape_rejected** (18-26, 15-23 of them at
  0.1 < occ <= 0.3). Per letter, the remaining loss is mostly `I` (recall 0.92-0.94 all):
  likely because a partly covered I (a plain bar) is hard to tell from a paddle-cut fragment
  and gets rejected (not verified per case; see the contact sheet).
- False positives 20-27 -> 0-1; precision >= 0.998.
- Position / angle accuracy on visible letters is unchanged or better (classic pos p95
  3.39 -> 0.61 mm); cnn "all" pos / angle p95 got slightly worse because the newly recovered
  occluded letters are harder (they were simply missing before).
- Held-out check (regression test set, seed 7, never used for tuning): visible recall 1.000 for
  all three methods, all recall 0.94 / 0.94 / 0.94 (was 0.67 / 0.68 / 0.72), so the gain is not
  specific to synth_val. Floors in test_recall_regression.py were raised to 0.97 / 0.97 / 0.91.

Caveat: all of this is synthetic data from the same generator (same paddle / skin colours the
occluder mask was tuned on); re-check on Gazebo / real frames when available.
