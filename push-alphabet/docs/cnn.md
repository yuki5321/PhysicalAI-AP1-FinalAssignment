# CNN 知覚の約束（データ形式・切り出し・モデル入出力）

古典版（`pa_perception/classic.py`）と同じ `Detection` を返す CNN 版を作るための、担当間の取り決めです。座標・角度は [interfaces.md](interfaces.md) と同じ world 座標（m・rad）。

## 方式

1. 二値化して文字ごとのかたまりを取る（`ClassicDetector.segment` と同じ考え方）。
2. かたまりの重心を中心に、決まった大きさで切り出す（下の「切り出し」）。
3. 小さな CNN が、文字種・向き・中心のずれを出す。
4. 必要なら古典版の IoU 微調整を後段に付ける（`method: hybrid`）。

## データセット形式

```
<root>/
  meta.json        # 生成条件（source, seed, 件数, scene.yaml の camera 値, ランダム化の設定）
  labels.jsonl     # 1 行 1 フレーム
  frames/000000.png  # 640x640 BGR（scene.yaml の camera 解像度）
```

`labels.jsonl` の 1 行:

```json
{"frame": "frames/000000.png", "source": "synth", "letters": [{"letter": "U", "x": 0.012, "y": -0.031, "theta": 1.234}]}
```

- `source` は `synth`（`pa_perception.synth` ＋ ランダム化）か `gazebo`（Gazebo から真値付きで保存）。
- `theta` は対称性で丸めない生の値（rad）。
- train / val / test はディレクトリを分ける（例 `data/synth_train`、`data/synth_val`、`data/gazebo_test`）。`data/` は git に入れない。

## 切り出し（`pa_perception/cnn_crop.py`）

- 入力はグレースケール画像（uint8）と、切り出し中心の画素座標 `(u, v)`（OpenCV の列・行）。
- 中心から 192×192 px の窓を取り（はみ出しは 0 で埋める）、2 分の 1 に縮小して **96×96**、値は 0..1 の float32。テンソル形は `[1, 96, 96]`。
- 学習時は中心を ±8 px ずらす（推論時のかたまり重心の誤差を吸収するため）。

## モデル入出力（ONNX）

- ファイル: `ros2_ws/src/pa_perception/models/letter_cnn.onnx`（小さければリポジトリに入れる、目安 2 MB 以下）。opset 17。推論は `onnxruntime`（Docker は Python 3.10、numpy<1.25）。
- 入力 `input`: float32 `[N, 1, 96, 96]`。
- 出力（この順・この名前）:
  - `logits`: `[N, 4]`。並びは `I, U, Y, K`（`scene.yaml` の `letters` の順）。
  - `angle`: `[N, 2]` = `(cos φ, sin φ)`。`φ = θ × 360 / sym`（`sym` は `scene.yaml` の `symmetry`。I は 180° 対称なので φ = 2θ）。正規化は復号側で行う。
  - `offset`: `[N, 2]` = 切り出し中心（文字上面の高さで world に写した点）から文字中心までのずれ `(dx, dy)` を **cm** で。
- 復号（`cnn_crop.decode`）: 文字種 = argmax(logits)、確信度 = softmax の最大値、`θ = atan2(sin φ, cos φ) × sym / 360`、位置 = 切り出し中心の world 座標 + offset/100。

## 評価

- `tools/eval_perception.py` が同じデータセットで古典版・CNN 版・hybrid を比べ、文字種の正解率、位置誤差（平均・95%・最大 mm）、角度誤差（対称性込み、平均・95%・最大 °）、1 フレームの処理時間を表にする。
- 合成データだけで学習し、合成の val と、取れれば Gazebo の画像で評価する。

## 前段の分割（`pa_perception/segmentation.py`）

古典版・CNN 版の両方が使う、文字のかたまりを取る処理の約束です（取りこぼし改善で追加）。

- `segment_letters(img_bgr, params=None, scene=None, cam=None) -> Segmentation`。入力は必ず BGR（色を使うため、グレースケールに変換してから渡さない）。
- `Segmentation` の中身:
  - `fg`: uint8 0/255。文字の候補画素（明るく、彩度が低い。ヘラや手の色は含めない）。
  - `occluder`: uint8 0/255。文字を隠しうるもの（オレンジのヘラ、肌色の手など彩度の高い明るい物体）。
  - `blobs`: 文字 1 個ずつに分けたかたまりのリスト。各要素は `label`（`labels` 画像での番号）、`centroid_px` (u, v)、`area_px`、`bbox`、`touches_occluder`（かたまりの縁が occluder に接している）、`split_from`（くっついたかたまりを分けたときの元の番号、分けていなければ None）。
  - `labels`: int32 のラベル画像（0 = 背景）。
  - `threshold`: 使った明るさの閾値。
- 照合側は、occluder の画素を「見えていない」として扱う（IoU の計算から外す、面積の下限を下げる）。
