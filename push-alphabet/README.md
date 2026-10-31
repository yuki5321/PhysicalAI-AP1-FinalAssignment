# Push-Alphabet（シミュレーション版）

崩されても自分で直す、雪の文字ロボ。真上カメラで文字ブロック（Y・U・K・I）を見て、台の四辺にある4本のヘラ（サーボ相当）で押し、崩されても「YUKI」を並べ直します。Physical AI 応用1講座の最終課題として ROS 2 Humble + Gazebo で作り、その後に実機（Arduino + SG90）へ移します。

> 開発中です。現在の進み具合は下の「状態」を見てください。計画の全体は [docs/plan.md](docs/plan.md) にあります。

## 仕組み

```
真上カメラ ─▶ 知覚（文字認識・姿勢推定） ─▶ プランナー（どのヘラを何度振るか） ─▶ 4本のヘラ
Webカメラ ─▶ MediaPipe Hands ─▶ シミュ内の「手」で文字を崩す／手がある間は待つ
```

- プランナーは、2D 押しモデル（pymunk、Push-T と同じ考え方）で候補の押し方を先読みし、目標の位置と向きに一番近づく押し方を選びます。
- 寸法やヘラの配置は [`ros2_ws/src/pa_core/pa_core/scene.yaml`](ros2_ws/src/pa_core/pa_core/scene.yaml) に一か所でまとめ、Gazebo・2D モデル・知覚で共有します。

## 構成

| パス | 内容 |
| --- | --- |
| `ros2_ws/src/pa_core` | 共通ライブラリ：文字形状、2D 押しモデル、プランナー（ROS 非依存） |
| `ros2_ws/src/pa_interfaces` | メッセージ定義（LetterState、PushCommand、HandState） |
| `ros2_ws/src/pa_description` | `scene.yaml` から台・ヘラ・真上カメラの URDF を生成、文字・手のモデル |
| `ros2_ws/src/pa_gazebo` | world、launch、ヘラ制御（`push_executor`）、`scene_manager`、`hand_actor` |
| `ros2_ws/src/pa_perception` | 真上カメラ → 文字の種類と姿勢（古典 CV / CNN / hybrid を `method` で切替、合成画像レンダラ付き） |
| `ros2_ws/src/pa_planner` | 状態機械（待機 → 手が消えるのを待つ → 直す → 次の文字） |
| `host/hand_bridge.py` | Web カメラ + MediaPipe Hands → rosbridge で `/hand/state`（[host/README.md](host/README.md)） |
| `docker/` | 講義イメージ + 追加パッケージ（[docker/README.md](docker/README.md)） |
| `tools/sim2d_closed_loop.py` | 2D モデルだけでプランナーの閉ループ性能を測るスクリプト |
| `tools/gen_letter_models.py` | 文字ブロックの SDF を `scene.yaml` から再生成 |
| `tools/gen_dataset.py` / `train_cnn.py` / `eval_perception.py` | CNN 用データ生成（合成・Gazebo）/ 学習と ONNX 書き出し / 知覚の比較評価（[docs/cnn.md](docs/cnn.md)） |
| `docs/plan.md` / `docs/interfaces.md` | 実装計画 / ノード間のトピック・メッセージの約束 |

## 2D モデルでの確認（ホスト）

```bash
uv venv -p 3.12 .venv-host
uv pip install -p .venv-host pymunk numpy pyyaml opencv-python
PYTHONPATH=ros2_ws/src/pa_core .venv-host/Scripts/python tools/sim2d_closed_loop.py --trials 5
```

2D モデルでの閉ループ成功率（`--trials 20`、seed 0、30 回以内）: I 17/20、U 16/20、Y 19/20、K 16/20（調整前は I 5、U 17、Y 15、K 9）。

## Gazebo シミュレーション（Docker 内）

```bash
cd /ws && colcon build --symlink-install --base-paths src && source install/setup.bash
ros2 launch pa_gazebo sim.launch.py            # gui:=false seed:=0 scene_manager:=false hand_actor:=false
ros2 run pa_perception letter_detector
ros2 launch pa_planner planner.launch.py
```

- 台・壁・4本のヘラ・真上カメラは `pa_description/urdf_gen.py` が `scene.yaml` から起動時に生成します。
- ヘラは gazebo_ros2_control（速度インターフェース + JointTrajectoryController）。単体で振るには `ros2 topic pub --once /push_cmd pa_interfaces/msg/PushCommand "{id: 1, side: 0, angle_deg: 60}"`。
- `scene_manager` が文字を `sequence` 順に崩した姿勢で出し、そろうとカメラ外の展示棚へ `display_word` 順に並べます。4文字そろうと雪が降ります。

## テスト（ROS なし）

```bash
pip install pymunk numpy pyyaml opencv-python-headless pytest
for d in ros2_ws/src/pa_core ros2_ws/src/pa_perception ros2_ws/src/pa_planner ros2_ws/src/pa_gazebo ros2_ws/src/pa_description host; do
  PYTHONPATH=ros2_ws/src/pa_core:$d python -m pytest -q $d
done
```

## 状態

- [x] 共有シーン定義、文字形状、2D 押しモデル、先読みプランナー
- [x] メッセージ定義・ノード間インターフェース
- [ ] Docker 環境（Dockerfile・compose は作成済み、ビルド未確認）
- [ ] Gazebo の台・ヘラ・カメラ・文字（作成済み、Gazebo での動作未確認）
- [ ] 知覚・プランナーの ROS 2 ノード（作成済み・単体テスト済み、ROS での結合未確認）
- [ ] 手検出による崩し、YUKI 完成演出（作成済み、動作未確認）
- [ ] CNN 知覚と成功率評価（CNN は合成データで学習・評価済み、Gazebo 画像と押し成功率は未評価）

### 知覚の比較（合成 val 1000 枚、遮蔽 10% 以下の文字 1110 個）

| 方式 | 検出率 | 適合率 | 位置誤差 95% | 角度誤差 95% | 1 フレーム |
| --- | --- | --- | --- | --- | --- |
| classic | 99.7% | 99.9% | 0.61 mm | 0.90° | 28 ms |
| cnn | 99.7% | 100% | 0.90 mm | 4.16° | 19 ms |
| hybrid（CNN + 古典の向き合わせ） | 99.6% | 100% | 0.90 mm | 0.46° | 30 ms |

強く隠れた文字も含めた全 1644 個での検出率は約 97%（改善前 69〜75%）。詳細と取りこぼしの内訳は [docs/perception_eval.md](docs/perception_eval.md)。

- [ ] 動画・レポート

## ライセンス

MIT
