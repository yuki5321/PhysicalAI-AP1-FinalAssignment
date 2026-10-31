# Push-Alphabet シミュレーション版（講座最終課題）実装計画

## Context

- 講座最終課題（締切 11/2 10:00、計画では 11/1 中に提出）は「講義で使った ROS / Gazebo / Genesis 等のシミュレータ」での提出を想定。AI 使用が必須（推論のみ可）。提出物は動画 1〜3 分（YouTube、タイトル `PAI最終課題_<omnicampusアカウント>`）、public GitHub、レポート PDF（`説明資料_PAI最終課題_<アカウント>.pdf`）。個人情報（顔・氏名）は一切出さない。
- ユーザーの判断: 開発計画書の「崩されても自分で直す雪の文字ロボ」を、**ROS 2 Humble + Gazebo** で先に作って講座に提出し、その後に実機（Arduino Uno R4 + SG90×4）に進む。
- 決定事項: 文字は **YUKI の4文字を I→U→Y→K の順に1文字ずつ直す**（時間切れなら I・U で提出）。AI は **(1) 真上カメラ画像からの文字認識・姿勢推定** と **(2) PC の Web カメラ + MediaPipe Hands で自分の手を検出し、シミュ内の「手」で文字を崩す／手がある間はロボが待つ**。模倣学習は今回入れない（Final Stage 構想としてレポートに書く）。
- 実機移行を見据え、ノード構成とトピック／指令インターフェースを実機と共通にする（Gazebo → usb_cam、ros2_control → Arduino シリアルブリッジに差し替えるだけにする）。

## 環境

- 講義第4・5回の Docker イメージ `airobotbook/ros2-desktop-ai-robot-book-humble:latest`（noVNC: localhost:6080、rosbridge 用に 9090 を公開）をベースに、プロジェクト用 `Dockerfile` を作る（不足分: `ros-humble-rosbridge-server`、`gazebo_ros2_control` / `ros2_controllers`、`python3-opencv`、CPU 版 torch）。
- 最初にコンテナ内で Gazebo のバージョン（Classic 11 か Ignition/Fortress か）と上記パッケージの有無を確認し、Gazebo Classic 前提で進める（CRANE+ V2 演習と同じ）。違えば world/plugin 記法だけ合わせる。
- リポジトリは ASCII パスに新規作成: `C:\dev\push-alphabet`（日本語・空白入りパスは Docker マウントと colcon で問題が出やすいため）。コンテナには `/ws/src/push-alphabet` としてマウント。
- ホスト（Windows）側: Python + `mediapipe` + `opencv-python` + `roslibpy`（手検出だけはホストで動かす。Docker から Windows の Web カメラを使うのは手間が大きいため）。
- GPU: RTX 4060 8GB / RAM 32GB。CNN 学習はホストの GPU、推論はコンテナ CPU で十分。

## リポジトリ構成

```
push-alphabet/
├── docker/Dockerfile, compose.yaml          # 講義イメージ + 追加パッケージ、6080/9090 公開
├── ros2_ws/src/
│   ├── pa_interfaces/        # msg: LetterState(letter, x, y, theta, conf), PushCommand(side, angle_deg), HandState(present, x, y)
│   ├── pa_description/       # 台(20cm角)・4方向プッシャー・真上カメラの xacro、文字ブロック SDF
│   ├── pa_gazebo/            # world、launch、ros2_control 設定、hand_actor / scene_manager ノード
│   ├── pa_perception/        # camera → LetterState（古典CV と CNN を切替可）
│   └── pa_planner/           # 状態機械 + プッシャー選択 → PushCommand
├── host/hand_bridge.py       # Webカメラ → MediaPipe Hands → roslibpy で /hand/state を publish
├── tools/
│   ├── gen_letter_models.py  # Y/U/K/I を箱の組み合わせで SDF 生成（実機のスチレンボード切り出し寸法と共通）
│   ├── gen_dataset.py        # Gazebo でランダム配置→画像＋真値姿勢を自動保存
│   ├── train_cnn.py          # 文字分類＋姿勢(sinθ,cosθ)回帰の小型CNN
│   └── eval_recovery.py      # ランダム崩し N 回の成功率・押し回数・所要時間を CSV 出力
├── docs/report/              # レポート原稿・図
├── README.md, LICENSE (MIT)
```

## 主要コンポーネント

1. **シミュ世界（pa_description / pa_gazebo）**
   - 約20cm角の台＋外周の低い壁。四辺中央に SG90 相当の**回転ヘラ**（revolute、位置制御、角度で押し量が決まる）を置き、実機と同じ「決まった角度まで振って押して戻す」動作にする。ヘラの回転中心は辺の中央からずらし、オフセット接触で文字を回転させられるようにする。
   - 文字ブロックは厚さ 5mm 相当の箱の組み合わせ（I=1枚、U/Y/K=3枚）、軽量・低摩擦（スチレンボード＋底テープ相当）。
   - 真上カメラ（gazebo_ros_camera、640×480）。
   - `scene_manager`: 文字の生成、そろった文字を「展示エリア」へ移す（人の役を代行）、次の文字を出す、4文字そろったら白い球を降らせる“雪”演出＋ライト色変更（LED 演出の代わり）。
   - `hand_actor`: `/hand/state` を受けて、台の上の運動学的な「手」モデルを `set_entity_state` で動かし、文字を物理的に崩す。

2. **知覚（pa_perception）— AI その1**
   - 古典版: 背景差分/二値化 → 輪郭 → 重心・最小外接矩形・モーメントで位置と向き、テンプレートで文字種（ベースライン兼フォールバック）。
   - CNN 版: `gen_dataset.py` で Gazebo から自動ラベル付き画像を数千枚生成 → `train_cnn.py` で文字分類＋向き回帰（I は 180° 対称を考慮）→ ONNX/torch でノード内推論。レポートで古典版との精度比較を示す。実機では同じネットを実写で追加学習する前提。

3. **手検出（host/hand_bridge.py）— AI その2**
   - MediaPipe Hands で手の有無・掌中心を取得し、画面座標→台座標に写像して rosbridge（ws://localhost:9090）経由で `/hand/state` に送る。映像は手元だけを映すよう画角を決める（顔を映さない）。

4. **判断（pa_planner）**
   - 状態機械: 待機 → 崩れ検知（位置/角度の閾値超え）→ 手が消えて 1 秒待つ → 直す（撮る→選ぶ→押す→待つ を収束まで繰り返し）→ そろった → 次の文字 → 4文字で完成。
   - プッシャー選択はルールベース: 位置誤差が大きい軸の反対側から押す、角度誤差はオフセット接触するヘラと角度を選ぶ、最後に対向ヘラで挟み込んで位置と向きをそろえる（センサレスな squeeze 整列）。押し角度は誤差に比例させ、上限で刻む。各パラメータはシミュで調整。
   - 出力 `PushCommand(side, angle)` は実機の Arduino シリアル指令とそのまま対応させる。

## 実装順とスケジュール（今日 10/22）

| 日付 | 完了の目安 |
|---|---|
| 10/22–23 | Docker 環境確認・構築、台・ヘラ・カメラ・I ブロックが Gazebo に出て、4本のヘラを指令で振れる |
| 10/24–25 | 古典知覚＋プランナーで、ランダムに置いた I・U を目標位置にそろえられる |
| 10/26–27 | Y・K 対応、hand_bridge と hand_actor で手で崩す→待つ→直すの一連が動く、scene_manager で YUKI 完成演出 |
| 10/28–29 | CNN 知覚（データ生成・学習・比較）、`eval_recovery.py` で文字ごとの成功率を測定しパラメータ調整 |
| 10/30–31 | 動画撮影・編集（Gazebo 画面＋手元カメラのワイプ、倍速部分は明記）、README、レポート PDF、GitHub public 化 |
| 11/1 | 提出（omnicampus：PDF・GitHub URL、YouTube URL）。11/2 朝は予備 |

Y・K が 10/27 時点で不安定なら、I・U＋手による崩しで動画を作り、Y・K は「挑戦中」としてレポートに書く。

## 提出物とレポートの骨子

- 動画（約2〜3分）: 冒頭に YUKI 完成の雪演出 → 構成図 → 手で崩す→手を引く→ロボが直す を文字ごとに（後半は倍速、表示あり）→ 実機への展開予告。
- レポート: 目的（Push-T を机上の文字並べに拡張）、システム構成図、AI の使い方（CNN 姿勢推定・MediaPipe 手検出）、工夫（squeeze 整列、オフセット接触による回転、シミュでの自動データ生成、実機と共通のインターフェース）、定量評価（成功率・押し回数）、うまくいかなかった点と考察、実機・1st Stage・Final Stage への計画。
- 個人情報チェック: 動画・コード・コミット履歴・GitHub プロフィールに氏名やメールが出ないこと（git の author 設定も確認）。

## 提出後（実機フェーズ）への引き継ぎ

- `pa_gazebo` の代わりに `usb_cam` ノードと `pa_arduino_bridge`（PushCommand → シリアル、Arduino スケッチは角度指令を受けてサーボを振って戻す）を追加するだけで、知覚・判断・手検出は流用する。CNN は実写画像で追加学習する。この部分は講座提出後に別途計画する。

## 検証方法

- 環境: `docker compose up` → noVNC で Gazebo 起動、`ros2 topic list` に camera / letter_state / push_cmd / hand/state が出る。
- 単体: `colcon build && colcon test`（知覚の姿勢推定を既知姿勢の生成画像で誤差確認、プランナーの選択ロジックをテスト）。
- 結合: `eval_recovery.py` で各文字 20 回ランダムに崩し、成功率（目標 I・U 90%以上、Y・K 70%以上）と平均押し回数を CSV に記録。
- 手: ホストで `python host/hand_bridge.py` を起動し、手をかざすとシミュ内の手が動いて文字が崩れ、手を引くとロボが直し始めることを目視確認。
- 提出前: 動画の長さ 1〜3 分・倍速表記・タイトル、PDF ファイル名、リポジトリが public であることを確認。
