# host/ — 手検出ブリッジ（Windows ホスト）

Web カメラ → MediaPipe Hands → 掌中心を台の座標に写像 → rosbridge（`ws://localhost:9090`）で `/hand/state`（`pa_interfaces/HandState`）を約 15 Hz で publish します。

```bash
pip install -r host/requirements.txt
python host/hand_bridge.py --no-ros --show-camera   # まず画角だけ確認（顔が映らないこと）
python host/hand_bridge.py --crop 0.2 0.1 0.8 0.9 --mirror
python -m pytest host                                # 写像の単体テスト（mediapipe 不要）
```

- **カメラは机の上の手元だけを映す向きにしてください（顔を映さない）。** 映像はメモリ上で処理するだけで保存しません。プレビューは既定で黒背景に手の骨格と台の枠だけを描きます（`--show-camera` で実画像）。
- 校正: `--crop U0 V0 U1 V1`（画像の正規化座標、u 右・v 下）の長方形を、`scene.yaml` の `workspace.table_half_size` の正方形に対応させます。画像の上 = world +y。`--mirror` で左右反転（プレビューで `m` キーでも切替）、`--rotate` で 90° 単位の回転。長方形の外の手は「台の上に手はない」扱いです。
- 出現は 2 フレーム連続、消失は 5 フレーム連続で切り替え、位置は指数平滑（`--alpha`）します。終了時は `present=false` を送ってロボを解放します。
