# Docker 環境

講義第4・5回の `airobotbook/ros2-desktop-ai-robot-book-humble:latest`（Ubuntu 22.04 + ROS 2 Humble + Gazebo Classic 11 + noVNC）に、このプロジェクトで使うパッケージを足したイメージです。

## 追加しているもの

| 種類 | 内容 |
| --- | --- |
| apt | `ros-humble-rosbridge-server`、`ros-humble-gazebo-ros-pkgs`、`ros-humble-gazebo-ros2-control`、`ros-humble-ros2-control`、`ros-humble-ros2-controllers`、`ros-humble-cv-bridge`、`ros-humble-xacro`、`ros-humble-robot-state-publisher`、`python3-opencv`、`python3-pytest`、`python3-colcon-common-extensions` |
| pip | `pymunk`（2D 押しモデル）、CPU 版 `torch` / `torchvision`（CNN 推論） |
| その他 | 期限切れの ROS apt 鍵の更新、`/etc/bash.bashrc` で ROS とワークスペースを自動 `source` |

`numpy` はベースイメージと同じく 1.25 未満に固定しています（`cv_bridge` などがシステムの numpy で作られているため）。

## ディレクトリ構成

```
/ws                         ← colcon ワークスペース（ここで colcon build）
├── src/push-alphabet/      ← このリポジトリ（ホストからマウント）
│   └── ros2_ws/src/pa_*    ← パッケージ本体（colcon が再帰的に見つける）
├── build/  install/  log/  ← 名前付きボリューム（コンテナを作り直しても残る）
```

リポジトリの `ros2_ws/` は「パッケージ置き場」で、ビルド成果物はリポジトリの外（ボリューム）に置きます。Windows のマウント上でビルドすると遅く、`--symlink-install` のシンボリックリンクも作れないことがあるためです。

## 使い方

リポジトリのルートで実行します（Windows は PowerShell でも同じ）。

```bash
# ビルドと起動（初回はベースイメージの取得に時間がかかります）
docker compose -f docker/compose.yaml up -d --build

# ブラウザで http://localhost:6080 を開く（noVNC のデスクトップ）

# コンテナに入る（デスクトップのターミナルと同じ ubuntu ユーザー）
docker exec -it -u ubuntu push-alphabet bash
```

コンテナ内でのビルドとテスト:

```bash
cd /ws
colcon build --symlink-install --base-paths src
source install/setup.bash
colcon test --base-paths src --packages-select pa_core && colcon test-result --verbose
```

ROS なしで `pa_core` だけ試す場合:

```bash
cd /ws/src/push-alphabet
PYTHONPATH=ros2_ws/src/pa_core python3 -m pytest ros2_ws/src/pa_core
PYTHONPATH=ros2_ws/src/pa_core python3 tools/sim2d_closed_loop.py --trials 5
```

## ポート

| ホスト | 用途 |
| --- | --- |
| 6080 | noVNC（コンテナ内の 80 番） |
| 9090 | rosbridge（`ros2 launch rosbridge_server rosbridge_websocket_launch.xml`）。ホストの `host/hand_bridge.py` がつなぐ |

## 片付け

```bash
docker compose -f docker/compose.yaml down      # コンテナを止めて消す（ビルド成果物は残る）
docker compose -f docker/compose.yaml down -v   # ビルド成果物のボリュームも消す
```

## 確認状況

- ベースイメージを取得して中身を確認済み: Ubuntu 22.04、Python 3.10.12、Gazebo 11.10.2（Classic）、`gazebo_ros_pkgs` 入り、`numpy` 1.24.4、`ubuntu` ユーザー、noVNC はコンテナ内 80 番（supervisord の `websockify ... 80`）、ROS の apt 鍵は `/usr/share/keyrings/ros-archive-keyring.gpg`。
- `docker build` そのものは未確認です。作成時の環境のネットワーク制限で `packages.ros.org` と `download.pytorch.org` に届かなかったためです（`pymunk` の Python 3.10 用 wheel が取れることは確認済み）。初回ビルドで apt / pip が失敗したら、パッケージ名とバージョン指定を見直してください。
