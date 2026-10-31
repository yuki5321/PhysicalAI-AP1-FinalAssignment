# 手で崩したときの復帰成功率（2D モデル、知覚 truth / classic / hybrid）

`tools/eval_recovery.py --disturb hand` で、「手で払って崩す → 手が去るのを待つ → ヘラで押して戻す」を文字ごとに 20 回ずつ測った結果です。評価の約束は [recovery_eval.md](recovery_eval.md)、プランナーと押しモデルの改善の記録（知覚 truth のみ）は [recovery_results.md](recovery_results.md) にあります。この文書はそれに **画像の知覚（classic / hybrid）** を加え、teleport との比較、手が途中で戻ってくる場合（`--rehand-prob`）をまとめます。

> **これは 2D モデル（`tools/recovery_sim2d.py`、pymunk の `PushSim`）での結果で、Gazebo ではありません。** 世界とプランナーの予測が同じモデルなので、モデル誤差（摩擦・接触・ヘラの動きの違い）は含まれていません。画像の知覚は `pa_perception.synth` の合成画像で、Gazebo のレンダリングとも異なります。Gazebo での数字はこれより下がると考えてください。

測定したコード: コミット `8158ee4`（プランナー調整 `ecd2b52` と、手の崩しの改良を含む）。

## 結論

- **手で崩した場合（hand）は、3 つの知覚・2 seed のすべてで I・Y・K が目標に届きました。** U は hybrid の seed 1 だけ 17/20（85%、目標 90%）で 1 回足りず、ほかは 90〜95%。
- 合計（/80）: truth 76・75、classic 76・75、hybrid 73・74。**画像の知覚にしても成功率はほとんど下がりません**（truth との差は 0〜3 回で、20 回の試行の誤差の範囲）。
- **`pushed_with_hand`（手がある間、または手が去って `hand_clear_s` 未満で押した）は 0 件**（全 640 試行、`hand_violations` もすべて 0）。手が去ってから押し始めるまでの待ちは毎回 1.00 s（= `hand_clear_s`）で、手が途中で戻った試行では合計 2.0 s。
- **teleport の方が難しい**: 同じ seed 0 で truth 68、classic 65、hybrid 63 / 80。差のほとんどは U（teleport では 40〜65%）。teleport は向きを大きく崩す（角度のずれの中央値 78°）のに対し、手の払いは位置を大きく動かすが向きはあまり変えない（中央値 15°）。U は大きく回った状態から戻すのが苦手で、teleport ではこれが効きます。
- **手が復帰中に戻ってくる場合（`--rehand-prob 0.3`、hybrid）**: 70/80。手が戻らなかった 57 試行は 53 成功（戻らない hand_hybrid_s0 と同じ試行は結果も同じ）、戻った 23 試行は 17 成功。一時停止 → 再開そのものは正しく動きました（手がある間に押した回数 0、待ちは 2 回分）。失敗 6 件のうち 4 件は戻ってきた手が文字を作業領域の外まで払った `out_of_workspace` です。I は 17/20（85%）で目標に届かず。
- 失敗はほぼすべて `max_pushes`（30 回押しても目標の 8 mm・6° に入らない）。`lost`・`false_done`・`timeout`・`failed` は 0 件。

## 「手」の崩し（sim2d）とは

`--disturb hand` は、Gazebo の手モデル（`pa_description/models/hand`: 手のひら 1 枚 ＋ 指の箱）の当たり形状を、pymunk の運動学的な多角形として 2D 世界に入れ、Gazebo バックエンド・`hand_actor` と同じ軌道（`hand_logic.step_hand`、30 Hz）で動かします。

1. 退避位置から高い位置（z_up）で飛んでくる。
2. 文字の手前（文字の外接円 ＋ 手のひらの半幅 ＋ 6 mm）に降りる。向き（払う方向）はランダム、文字の中心から横に最大 ±20 mm ずらす（回転も起こす）。
3. 払う方向にまっすぐ滑り、文字の中心を 5〜40 mm 越えたところで上がる。速度は最大 0.15 m/s（`--hand-vmax`）。手の箱が文字の高さより下にある間だけ文字に当たる。
4. 上がって退避位置へ飛び去る。
5. 崩れていない（`is_disturbed` でない）か作業領域の外に出たら、同じ開始姿勢からやり直し（最大 4 回、`--hand-tries`）、それでも駄目なら teleport に切り替える。今回の 1 seed あたり 80 試行では、1 回目で崩れたのが 76〜78 回、2 回目が 2〜4 回で、teleport への切り替えは 0 回。

払っている間の流れは状態機械にそのまま流します。

- 手の有無は `/hand/state` と同じ意味で `on_hand(True/False)` を 30 Hz で送る（退避位置を離れてから上がり始めるまで「いる」）。
- 観測は 10 Hz。truth では文字の面積の半分より多くが手の下に隠れていれば見えない。classic / hybrid では合成画像に手（肌色の箱）を描いて検出器に通す。
- 状態機械は手が消えてから `hand_clear_s`（1.0 s）待って復帰を始める。手がある間・消えて 1.0 s 未満に押したら `pushed_with_hand` で失敗とする。
- `--rehand-prob p`: 確率 p で、復帰中の 1〜3 回目の押しの最中か直後の静定中に手がもう一度来て払う（一時停止 → 再開の確認）。

崩した直後の目標からのずれ（`disturb_dp_mm` / `disturb_da_deg`、中央値 [最小〜最大]）:

| 崩し | seed | 位置のずれ mm | 角度のずれ ° |
|---|---|---|---|
| hand | 0 | 95 [59〜130] | 15 [0〜60] |
| hand | 1 | 98 [60〜121] | 14 [0〜126] |
| teleport | 0 | 61 [7〜90] | 78 [2〜178] |

## 結果（各文字 20 回、押し 30 回まで、`--jobs 2`）

列: 成功数 / 試行数、目標達成、平均押し回数（成功時）、平均所要時間（sim 時間、成功時、手を待つ時間を含む）、平均手待ち（全試行）、失敗理由。目標は I・U 90%、Y・K 70%。

### hand、知覚 truth

| 文字 | seed | 成功 | 目標 | 平均押し | 平均時間 s | 手待ち s | 失敗理由 |
|---|---|---|---|---|---|---|---|
| I | 0 | 20/20 (100%) | ○ | 8.6 | 16.5 | 1.00 | - |
| U | 0 | 19/20 (95%) | ○ | 9.3 | 16.7 | 1.00 | max_pushes 1 |
| Y | 0 | 20/20 (100%) | ○ | 6.5 | 11.8 | 1.00 | - |
| K | 0 | 17/20 (85%) | ○ | 9.9 | 17.4 | 1.00 | max_pushes 3 |
| I | 1 | 20/20 (100%) | ○ | 10.4 | 20.4 | 1.00 | - |
| U | 1 | 18/20 (90%) | ○ | 10.4 | 18.4 | 1.00 | max_pushes 2 |
| Y | 1 | 20/20 (100%) | ○ | 7.5 | 13.7 | 1.00 | - |
| K | 1 | 17/20 (85%) | ○ | 8.9 | 15.5 | 1.00 | max_pushes 3 |

計: seed 0 = 76/80、seed 1 = 75/80（[recovery_results.md](recovery_results.md) の「後」と同じ数字）。

### hand、知覚 classic

| 文字 | seed | 成功 | 目標 | 平均押し | 平均時間 s | 手待ち s | 失敗理由 |
|---|---|---|---|---|---|---|---|
| I | 0 | 20/20 (100%) | ○ | 9.5 | 18.3 | 1.00 | - |
| U | 0 | 19/20 (95%) | ○ | 10.4 | 18.5 | 1.00 | max_pushes 1 |
| Y | 0 | 20/20 (100%) | ○ | 6.4 | 11.6 | 1.00 | - |
| K | 0 | 17/20 (85%) | ○ | 9.0 | 15.8 | 1.00 | max_pushes 3 |
| I | 1 | 19/20 (95%) | ○ | 8.2 | 15.9 | 1.00 | max_pushes 1 |
| U | 1 | 18/20 (90%) | ○ | 10.4 | 19.1 | 1.00 | max_pushes 2 |
| Y | 1 | 20/20 (100%) | ○ | 7.3 | 13.1 | 1.00 | - |
| K | 1 | 18/20 (90%) | ○ | 8.6 | 15.1 | 1.00 | max_pushes 2 |

計: seed 0 = 76/80、seed 1 = 75/80。

### hand、知覚 hybrid

| 文字 | seed | 成功 | 目標 | 平均押し | 平均時間 s | 手待ち s | 失敗理由 |
|---|---|---|---|---|---|---|---|
| I | 0 | 19/20 (95%) | ○ | 9.2 | 17.9 | 1.00 | max_pushes 1 |
| U | 0 | 18/20 (90%) | ○ | 9.2 | 17.0 | 1.00 | max_pushes 2 |
| Y | 0 | 20/20 (100%) | ○ | 8.1 | 14.4 | 1.00 | - |
| K | 0 | 16/20 (80%) | ○ | 10.7 | 19.1 | 1.00 | max_pushes 4 |
| I | 1 | 19/20 (95%) | ○ | 8.3 | 17.1 | 1.00 | max_pushes 1 |
| U | 1 | **17/20 (85%)** | × | 9.8 | 17.7 | 1.00 | max_pushes 3 |
| Y | 1 | 20/20 (100%) | ○ | 7.7 | 13.8 | 1.00 | - |
| K | 1 | 18/20 (90%) | ○ | 9.3 | 16.7 | 1.00 | max_pushes 1, out_of_workspace 1 |

計: seed 0 = 73/80、seed 1 = 74/80。

### hand と teleport の比較（seed 0、成功数 / 20）

| 文字 | 目標 | truth hand | truth teleport | classic hand | classic teleport | hybrid hand | hybrid teleport |
|---|---|---|---|---|---|---|---|
| I | 90% | 20 | 20 | 20 | 20 | 19 | 19 |
| U | 90% | 19 | **13** | 19 | **11** | 18 | **8** |
| Y | 70% | 20 | 20 | 20 | 20 | 20 | 20 |
| K | 70% | 17 | 15 | 17 | 14 | 16 | 16 |
| 計 /80 | | 76 | 68 | 76 | 65 | 73 | 63 |

teleport の詳細（seed 0）:

| 知覚 | 文字 | 成功 | 平均押し | 平均時間 s | 失敗理由 |
|---|---|---|---|---|---|
| truth | I / U / Y / K | 20 / 13 / 20 / 15 | 7.7 / 14.5 / 8.6 / 12.0 | 14.8 / 26.5 / 16.1 / 21.4 | U: max_pushes 7、K: max_pushes 5 |
| classic | I / U / Y / K | 20 / 11 / 20 / 14 | 7.2 / 20.2 / 8.4 / 11.4 | 14.4 / 36.0 / 15.6 / 20.0 | U: max_pushes 6・out_of_workspace 3、K: max_pushes 6 |
| hybrid | I / U / Y / K | 19 / 8 / 20 / 16 | 6.1 / 18.0 / 9.3 / 8.8 | 11.7 / 31.1 / 17.2 / 16.2 | I: max_pushes 1、U: max_pushes 11・out_of_workspace 1、K: max_pushes 4 |

teleport の U では知覚の差が出ます（truth 13 → classic 11 → hybrid 8）。大きく傾いた U は、truth でももともと押し回数上限で失敗しやすく（recovery_results.md「残っている失敗」）、画像の知覚が加わると角度の推定誤差でさらに往復が増えます。hand の崩しではこの場面がまれなので、差はほとんど出ません。

### 手が復帰中に戻ってくる場合（`--rehand-prob 0.3`、hybrid、seed 0）

| 文字 | 成功 | 目標 | 平均押し | 平均時間 s | 手待ち s（平均） | 失敗理由 |
|---|---|---|---|---|---|---|
| I | **17/20 (85%)** | × | 9.4 | 19.4 | 1.35 | max_pushes 2, out_of_workspace 1 |
| U | 18/20 (90%) | ○ | 10.9 | 21.4 | 1.25 | max_pushes 2 |
| Y | 20/20 (100%) | ○ | 7.8 | 14.8 | 1.15 | - |
| K | 15/20 (75%) | ○ | 11.5 | 21.9 | 1.40 | out_of_workspace 3, max_pushes 2 |

計 70/80（手が戻らない hand_hybrid_s0 は 73/80）。手が戻ってきた押しの回数ごと:

| 手が戻った押し | 試行 | 成功 | 失敗理由 | 手待ちの合計 s |
|---|---|---|---|---|
| なし | 57 | 53 | max_pushes 4 | 1.0 |
| 1 回目 | 7 | 5 | out_of_workspace 1, max_pushes 1 | 2.0 |
| 2 回目 | 8 | 7 | max_pushes 1 | 2.0 |
| 3 回目 | 8 | 5 | out_of_workspace 3 | 2.0 |

- 一時停止 → 再開は 23 試行すべてで正しく動いた（手がある間・去って 1.0 s 未満に押した回数 0、待ちは 2 回分の 2.0 s）。
- `out_of_workspace` 4 件はすべて手が戻ってきた試行で、1 件（I#01）は手が戻った 1 回目の押しの直後、2 件（K#08・K#09）は 3 回目の直後 1 回以内の押しで終わっており、**戻ってきた手が文字を台の端（壁の手前）まで払ったもの** と考えられます（2 回目の払いは「作業領域内に残る」払いを選びますが、選ぶのは押しの前の姿勢で、手は押しの途中か直後に着くため外れることがある）。人の手が台の外まで払った場合はロボットの責任ではないので、この 4 件は復帰の失敗というより崩しの側の結果です。

## 失敗の内訳と残っている失敗

hand 6 run（480 試行）の失敗は 31 件: `max_pushes` 30、`out_of_workspace` 1。`lost`・`false_done`・`timeout`・`failed`・`pushed_with_hand` は 0。

- **K の同じ試行が知覚によらず失敗**: seed 0 の K#08、seed 1 の K#15 は truth・classic・hybrid のすべてで、目標から約 13 cm・約 20° のところで 30 回押し切る。手の払いが文字を台の端近くまで動かした場面で、recovery_results.md の「四隅の死角」（どのヘラでも中央へ戻せない位置）に当たると考えられます。同様に seed 0 の K#09、seed 1 の U#7 も複数の知覚で失敗。
- **目標付近の停滞（K・U）**: 1〜4 cm・数〜20° で止まる。truth でも起きる（recovery_results.md と同じ）。
- **ぎりぎりで「そろった」と言えない**: classic の K#1（seed 0）は真値で 2.3 mm・5.8°（成功判定の 8 mm・6° の内側）で終わったが、状態機械の「そろった」判定は `done_margin=0.8`（6.4 mm・4.8°）なので止まらず、押し回数上限になった。classic の U#11（6.9 mm・2.4°）、U#1（5.1 mm・6.7°）も目標のすぐ外。誤完了（`false_done`）を防ぐ代わりに、知覚の誤差がある場合は目標付近で押し続ける回数が増えます。
- **hybrid seed 1 の K#16**: 22 回押したところで作業領域外（大きな押しで飛ばした）。hand 全体でこの 1 件のみ。
- **teleport の U**: 大きく傾いた U を回しきれない（truth でも 13/20）。画像の知覚では更に下がる（hybrid 8/20）。U の回転の手順か価値関数の導入が必要（recovery_results.md）。

## 再現

```bash
export PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception
E="python3 tools/eval_recovery.py --backend sim2d --letters IUYK --trials 20 --jobs 2"
mkdir -p data/recovery/hand
# 手で崩す: 知覚 3 種 × seed 2
for p in truth classic hybrid; do for s in 0 1; do
  $E --perception $p --seed $s --disturb hand --csv data/recovery/hand/hand_${p}_s$s.csv \
     > data/recovery/hand/hand_${p}_s$s.log
done; done
# 比較用の teleport（seed 0）
for p in truth classic hybrid; do
  $E --perception $p --seed 0 --disturb teleport --csv data/recovery/hand/teleport_${p}_s0.csv \
     > data/recovery/hand/teleport_${p}_s0.log
done
# 手が復帰中に戻ってくる
$E --perception hybrid --seed 0 --disturb hand --rehand-prob 0.3 \
   --csv data/recovery/hand/rehand03_hybrid_s0.csv > data/recovery/hand/rehand03_hybrid_s0.log
```

試行は (seed, 文字, 試行番号) から乱数を決めるので、`--jobs` を変えても結果は同じです。所要時間（このマシン、4 コアを 2 本の実行で共有、各 `--jobs 2`）: truth 約 3.5 分、classic・hybrid 約 8 分 / run、全 10 run で約 35 分。出力（CSV・ログ）は `data/recovery/hand/` で、git には入りません（`data/` は .gitignore）。
