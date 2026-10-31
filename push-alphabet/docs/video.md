# 提出動画の作り方（撮影・編集・アップロードの手順）

講座の提出動画（1〜3 分、YouTube）を作るための手順です。Windows ホスト + Docker コンテナを前提にしています。

- 提出の条件（[plan.md](plan.md)）: 長さ 1〜3 分、YouTube のタイトルは `PAI最終課題_<omnicampusアカウント>`、**顔・氏名などの個人情報は一切出さない**、倍速の部分は画面に倍速と書く。締切は 11/2 10:00（計画では 11/1 に提出）。
- 動画の流れ: 冒頭に YUKI 完成と雪 → タイトル → 構成図 → AI の説明 → 手で崩す → 手を引く → ロボが直す を文字ごとに（後半は倍速、表示あり）→ 結果 → 実機への予告。

## 使う道具

| 道具 | 役割 |
| --- | --- |
| `ros2 launch pa_gazebo demo.launch.py` | Gazebo のデモ一式（シミュ・知覚・プランナー・rosbridge・録画ノード） |
| `demo_recorder`（demo.launch.py の `record:=true`） | 真上カメラ映像に状態表示を重ねた MP4 を `videos/demo_<日時>.mp4` に保存 |
| `tools/record_gazebo_screen.sh` | コンテナのデスクトップ（Gazebo の 3D 画面）を録画 → `videos/screen_<日時>.mp4` |
| `host/hand_bridge.py` | Web カメラ → MediaPipe Hands → シミュ内の手 |
| OBS Studio（ホスト、任意） | 手元カメラのワイプ用の録画 |
| `tools/make_demo_video.py` | Gazebo なしで 2D モデルだけのデモ動画 `data/video/demo_2d.mp4` を作る（予備） |
| `tools/video_cards.py` | タイトル・構成図・AI 説明・結果・実機予告のカード画像 |
| `tools/assemble_video.py` | 編集リスト（YAML）から最終 MP4（1280×720、30 fps、H.264）を組み立てる |

`videos/` と `data/` は git に入りません（`*.mp4` と `data/` は `.gitignore` 済み）。

---

## 1. コンテナの準備（Windows・PowerShell）

リポジトリのルート（例 `C:\dev\push-alphabet`）で:

```powershell
docker compose -f docker/compose.yaml up -d --build     # 初回は時間がかかる
docker exec -it -u ubuntu push-alphabet bash             # コンテナに入る
```

コンテナ内:

```bash
cd /ws
colcon build --symlink-install --base-paths src
source install/setup.bash
which ffmpeg     # 録画に必要。無ければ: sudo apt-get update && sudo apt-get install -y ffmpeg x11-utils
```

ブラウザで http://localhost:6080 を開くと noVNC のデスクトップが出ます（Gazebo の画面はここに出ます）。

## 2. デモを起動する

コンテナ内（noVNC のデスクトップのターミナルでも `docker exec` でも可）:

```bash
ros2 launch pa_gazebo demo.launch.py gui:=true record:=true
```

- 引数の正確な名前と既定値は **`ros2_ws/src/pa_gazebo/launch/demo.launch.py` の先頭のコメントで確認してください**（`seed`、`method`、`out_dir`、`auto_hand` など。作成中のため変わることがあります）。`ros2 launch pa_gazebo demo.launch.py --show-args` でも一覧が出ます。
- 録画ファイルは既定で `/ws/src/push-alphabet/videos/`（= ホストのリポジトリの `videos\`）にできます。止めると H.264 に変換されるので、止めてから数秒〜数十秒待ってから使ってください。
- まずは録画なし（`record:=false`）で 1 回通して、4 文字そろって雪が降るまで動くことを確かめるのがおすすめです。

## 3. 手検出をつなぐ（Windows ホスト）

別の PowerShell で（初回のみ `pip install -r host/requirements.txt`）:

```powershell
python host/hand_bridge.py --no-ros --show-camera   # まず画角の確認だけ
python host/hand_bridge.py --crop 0.2 0.1 0.8 0.9 --mirror
```

- **カメラは机の上の手元だけを映す向きにする。顔・部屋・名前の書いた物（ノート、封筒、モニタの通知）が入らないこと。** 袖口や指輪など特定につながる物も外しておくと安心です。
- `--crop` と `--mirror` は、手を動かしたときにシミュ内の手が同じ向きに動くように調整します（[host/README.md](../host/README.md)）。
- プレビュー窓は既定で「黒背景に手の骨格だけ」を描きます。**ワイプにはこの骨格プレビューを使うのが一番安全**です（実際のカメラ映像が映りません）。

## 4. 録画する

録画は 3 種類あります。最低限 (a) があれば動画は作れます。

### (a) 真上カメラ + 状態表示（コンテナ内、自動）

`record:=true` で起動していれば自動で録画されます（`videos/demo_<日時>.mp4`、シミュ時間で再生されるので Gazebo が遅くても動画は等速）。途中で止める・始めるときは:

```bash
ros2 service list | grep set_recording         # サービス名を確認
ros2 service call /demo_recorder/set_recording std_srvs/srv/SetBool "{data: true}"   # 名前は上で確認
```

### (b) Gazebo の 3D 画面（コンテナ内）

```bash
cd /ws/src/push-alphabet
tools/record_gazebo_screen.sh -w        # Gazebo の窓をクリックして、その窓だけ録画。Ctrl-C で終了
tools/record_gazebo_screen.sh -t 60     # 画面全体を 60 秒
```

`videos/screen_<日時>.mp4` ができます。ブラウザ全体ではなくコンテナのデスクトップだけが録画されるので、ホストのタブやアカウント名は映りません。

### (c) 手元カメラのワイプ（ホスト、任意）

`hand_bridge.py` は映像を保存しません。ワイプが欲しいときは **OBS Studio** で録画します。

1. OBS の「ソース」に「ウィンドウキャプチャ」で `hand_bridge (q: quit, m: mirror)` の窓を追加（骨格だけのプレビュー。おすすめ）。実際の手の映像を使う場合は「映像キャプチャデバイス」で Web カメラを追加し、**手元だけになるように切り取る**（ソースを Alt キーを押しながらドラッグ）。
2. 「設定 → 出力」で録画フォーマットを mp4、「設定 → 映像」で 30 fps。
3. 録画開始 → (a) の録画が始まってから**手を一度はっきりかざして引く**（あとで (a) とタイミングを合わせる目印）→ 終わったら停止。
4. できたファイルを `videos\handcam_<日時>.mp4` など分かる名前でリポジトリの `videos\` に置く。
5. **マイク音声は入れない**（OBS の音声ミキサーでマイクとデスクトップ音声をミュート）。部屋の声や通知音が入るのを防ぐためです。

ワイプのタイミング合わせ: (a) の動画で手が最初に台に入る秒と、(c) で手が最初に映る秒を動画プレイヤーで調べ、編集リストの `pip.in` = (c) の秒 −（(a) の秒 − `in`）にします。

## 5. ファイルをホストに出す

リポジトリはコンテナにマウントされているので、`videos/` と `data/video/` にできたファイルはそのまま Windows の `C:\dev\push-alphabet\videos\` などに見えます。別の場所に保存した場合は:

```powershell
docker cp push-alphabet:/tmp/a.mp4 .\videos\a.mp4
```

## 6. 組み立てる（`tools/assemble_video.py`）

### 実行場所

- **ホスト（Windows）がおすすめ**: 日本語フォント（游ゴシック・メイリオ）が自動で見つかります。

  ```powershell
  winget install Gyan.FFmpeg            # ffmpeg（入れたら PowerShell を開き直す）
  pip install pillow pyyaml numpy        # host の venv でよい
  python tools/assemble_video.py --write-example b data/video/edit_b.yaml
  python tools/assemble_video.py data/video/edit_b.yaml --dry-run   # 長さだけ確認
  python tools/assemble_video.py data/video/edit_b.yaml
  ```

- コンテナでも動きます（`python3 tools/assemble_video.py ...`、Pillow と PyYAML が無ければ `pip install pillow pyyaml`）。日本語フォントが無いとカードとラベルが英語になります。日本語にしたい場合は Noto Sans CJK などのフォントファイルを `data/fonts/`（git 対象外）に置くか、環境変数 `PA_FONT` にフォントのパスを指定してください。`python3 tools/video_cards.py --list` で使われるフォントが分かります。

### 編集リストの書き方

`python tools/assemble_video.py --write-example a|b PATH` で下の例をファイルに書き出せます。

- `card:` カード（`title` `diagram` `ai` `results` `future` `text`）。`duration` 秒。`text` は `big:`（大きな 1 行）+ `heading:`、または `heading:` + `lines:`。`title` は `course:` で講座名の行を差し替え（**氏名は書かない**）。
- `clip:` 動画の一部。`in` / `out`（元の動画の秒。負の数は終わりからの秒）、`speed`（2 なら 2 倍速。**1 以外なら右上に「×2 倍速」が自動で出る**。`speed_label: false` で消す）、`caption` / `captions`（`{text, start, end}`、出力側の秒）、`crop: [x, y, w, h]`。
- `pip:` ワイプ。`path`、`in`、`corner`（`tl` `tr` `bl` `br`）、`width`、`radius`、`border`、`label`。速さは本体と同じ（`speed` で変更可）。
- 全体: `output`、`lang`（`auto` / `ja` / `en`）、`fade`（各区切りの暗転秒、0 で切り替えのみ）、`bgm`（音楽ファイル）。
- オプション: `--silent`（無音の音声トラックを付ける）、`--bgm FILE`（BGM をループ・最後にフェードアウト。**著作権の問題がない音源だけ**）、`--strict`（60〜180 秒を外れたらエラー）、`--preset ultrafast`（試し書き用）。
- 合計の長さを毎回表示し、60 秒未満・180 秒超なら `WARNING` を出します。出力はメタデータを消してあります。

#### (A) Gazebo 版

`XXXX` を実際のファイル名に、`in` / `out` を実際の秒に書き換えます（動画プレイヤーで各文字が始まる秒を調べる）。

```yaml
# (A) Gazebo version. Replace the XXXX file names and the in/out times
# (open the MP4 in a player and note the seconds where each letter starts).
output: data/video/PAI_final.mp4
lang: ja
fade: 0.3
segments:
  # 1. opening: YUKI complete + snow (the last seconds of the recorder MP4)
  - clip: videos/demo_XXXX.mp4
    in: -8                     # negative = seconds from the end
    caption: "YUKI 完成！ 崩されても自分で直す"
  - card: title
    duration: 5
  - card: diagram
    duration: 9
  - card: ai
    duration: 8
  # 2. Gazebo 3D view (screen capture), short
  - clip: videos/screen_XXXX.mp4
    in: 5
    out: 15
    caption: "Gazebo（ROS 2 Humble）: 台・4本のヘラ・真上カメラ"
  # 3. letter I in real time, with the hand-camera wipe (hands only!)
  - clip: videos/demo_XXXX.mp4
    in: 0
    out: 30
    captions:
      - {text: "手で崩す（MediaPipe Hands）", start: 0, end: 8}
      - {text: "手がある間は待つ → 手を引くと直し始める", start: 8, end: 16}
      - {text: "押す → 見る → 押す… でそろえる", start: 16, end: 30}
    pip:
      path: videos/handcam_XXXX.mp4
      in: 0                    # handcam second that matches demo second `in` above
      corner: br
      width: 360
      label: "手元カメラ（手だけ）"
  # 4. U, Y, K faster (a "x2 倍速" label is added automatically)
  - clip: videos/demo_XXXX.mp4
    in: 30
    out: 70
    speed: 2
    caption: "U"
    pip: {path: videos/handcam_XXXX.mp4, in: 30, corner: br, width: 300, label: "手元カメラ"}
  - clip: videos/demo_XXXX.mp4
    in: 70
    out: 150
    speed: 2
    caption: "Y・K"
  - card: results
    duration: 9
  - card: future
    duration: 6
```

ワイプを使わない場合は `pip:` の行を消すだけです。録画が 1 本にまとまらなかった場合は、`clip:` に別のファイルを書けば並べられます。

#### (B) 予備: 2D デモだけで作る版（コンテナだけで完結）

Gazebo が動かないときの予備です。`tools/make_demo_video.py` で作った `data/video/demo_2d.mp4` とカードだけで組み立てます。demo_2d.mp4 には倍速部分の表示がすでに入っているので、ここでは等速（`speed` なし）で使います（重ねて速くすると表示と食い違うため）。

```bash
PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception \
    python3 tools/make_demo_video.py --out data/video/demo_2d.mp4     # 引数は make_demo_video.py の先頭を参照
python3 tools/assemble_video.py --write-example b data/video/edit_b.yaml
python3 tools/assemble_video.py data/video/edit_b.yaml               # -> data/video/PAI_final_fallback.mp4
```

```yaml
# (B) fallback built only from the 2D demo (tools/make_demo_video.py) + cards.
# demo_2d.mp4 already shows its own "x2" labels on sped-up parts, so it is used
# at speed 1 here (do not speed it up again, or the labels would be wrong).
output: data/video/PAI_final_fallback.mp4
lang: ja
fade: 0.3
segments:
  - clip: data/video/demo_2d.mp4
    in: -7                     # opening: YUKI complete + snow (last 7 s; the clip shows "YUKI 完成！" itself)
  - card: title
    duration: 5
  - card: diagram
    duration: 9
  - card: ai
    duration: 8
  - card: text
    big: "手で崩す → 手を引く → ロボが直す"
    heading: "2D 押しモデル（pymunk）でのシミュレーション。倍速部分は右上に表示"
    duration: 4
  - clip: data/video/demo_2d.mp4
  - card: results
    duration: 9
  - card: future
    duration: 6
```

(B) で提出する場合は、講座の条件（講義で使ったシミュレータでの提出）に合うか、レポートで「Gazebo 版は動作確認中、動画は同じ判断ロジックを 2D 押しモデルで動かしたもの」と正直に書いてください。Gazebo の画面が少しでも撮れていれば、(B) に (b) の画面録画を 10 秒ほど足すだけでも説得力が上がります。

### カードの確認

```bash
python3 tools/video_cards.py --out data/video/cards        # PNG を書き出して目で確認
python3 tools/video_cards.py --list                        # フォントと、結果カードに使う数値
```

- 結果カードの「崩し → 復帰」の数値は `docs/recovery_results.md` / `docs/recovery_hand.md` の表（`I U Y K` の列に `17/20` の形）があればそこから、無ければ `data/recovery/**/*.csv`（`eval_recovery.py` の出力、`smoke` は除く）から取ります。2D モデル（sim2d）の数値には「2D モデルでの評価」と注記が出ます。Gazebo で測り直したら CSV を `data/recovery/` に置けば、自動で新しいほうが使われます。
- 知覚の表は README の「知覚の比較」の表から取ります。

### できあがりの確認

```bash
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate data/video/PAI_final.mp4
ffmpeg -i data/video/PAI_final.mp4 -vf "fps=1/10,scale=320:-1,tile=4x4" -frames:v 1 data/video/sheet.png   # 10 秒ごとの一覧
```

最後に **必ず最初から最後まで 1 回通して見る**（倍速表示、字幕の誤字、個人情報の映り込み）。

---

## 7. 個人情報のチェックリスト

提出前に全部チェックしてください。

- [ ] **動画**: 顔・氏名・メールアドレス・学籍番号・住所が分かる物が映っていない。手元カメラに部屋・書類・モニタの通知が映っていない。ワイプは骨格プレビューか手だけ。
- [ ] **音声**: マイク音声が入っていない（`--silent` か BGM のみ）。
- [ ] **カード**: タイトルカードの講座名の行に氏名を入れていない（`course:` を使う場合も）。
- [ ] **ファイルのメタデータ**: `assemble_video.py` の出力はメタデータを消してあります。他のツールで書き出した場合は:

  ```bash
  ffmpeg -i in.mp4 -map_metadata -1 -map_chapters -1 -c copy out.mp4
  ffprobe -v error -show_entries format_tags:stream_tags -of default out.mp4   # encoder / handler_name 程度なら OK（title・artist・comment・location などが無いこと）
  ```

  Windows のエクスプローラーで「プロパティ → 詳細」に作成者などが入っていないかも見ておく。
- [ ] **ファイル名**: アップロードするファイル名に氏名を入れない（YouTube で公開されることはないが念のため）。
- [ ] **git の作者情報**: コミットに本名・個人のメールが入っていないか。

  ```bash
  git log --format='%an <%ae> | %cn <%ce>' | sort -u
  git config user.name; git config user.email
  ```

  本名が出ていたら、今後は `git config user.name "<ハンドル名>"`、`git config user.email "<ID>+<ユーザー名>@users.noreply.github.com"`（GitHub の Settings → Emails の noreply アドレス）にする。過去のコミットを書き換える場合は push 済みの履歴の書き換えになるので、やるなら public にする前に。
- [ ] **GitHub**: プロフィールの Name・Bio・アイコンに本名や顔写真がない。README・docs・コードのコメントに氏名がない（`git grep -n "<本名>"` で確認）。
- [ ] **YouTube**: チャンネル名・ハンドル・アイコンに本名や顔写真がない（必要なら別のチャンネルを作る）。動画の説明欄・タグにも氏名を書かない。

## 8. YouTube へのアップロード

1. YouTube Studio → 作成 → 動画をアップロード → `data/video/PAI_final.mp4`（または `PAI_final_fallback.mp4`）。
2. **タイトル: `PAI最終課題_<omnicampusアカウント>`**（`<...>` を自分のアカウント名に置き換える。氏名ではなくアカウント名）。
3. 説明: プロジェクトの 1〜2 行の説明と GitHub の URL、「倍速部分は画面右上に表示」。
4. 「子ども向けではない」を選ぶ。
5. 公開設定（公開 / 限定公開）は **講座の提出要項の指示に従ってください**。限定公開にする場合は URL を知っている人だけが見られます。
6. 公開後、ログアウトした状態（シークレットウィンドウ）で URL を開き、再生できること・チャンネル名に個人情報がないことを確認。
7. omnicampus に YouTube の URL を提出（レポート PDF・GitHub URL と一緒に）。

## 9. スケジュール（10/29〜11/1）

| 日付 | やること | 予備 (B) への切り替え判断 |
| --- | --- | --- |
| 10/29 | コンテナのビルド、`colcon build`、`demo.launch.py` を録画なしで 1 回通す。まず予備 (B) を作っておく（`make_demo_video.py` → `assemble_video.py edit_b.yaml`）| — |
| 10/30 | hand_bridge をつないで手で崩す → 直すを確認。(a) 真上カメラ録画、(b) Gazebo 画面録画、(c) OBS で手元ワイプを撮る | 10/30 夜に Gazebo で 1 文字（I）も直せない場合は (B) で行く |
| 10/31 | 編集リスト (A) を書いて組み立て、通して見て直す。レポート PDF の作業 | (A) の素材が足りなければ (B) + 撮れた Gazebo 画面を短く挿入 |
| 11/1 | 個人情報チェック → YouTube アップロード → omnicampus に提出 | 11/1 の昼までに (A) ができていなければ (B) で提出 |
| 11/2 朝 | 予備（締切 10:00） | — |

予備 (B) はコンテナ内だけで作れるので、10/29 のうちに一度作っておけば、最悪でも提出できる状態になります。
