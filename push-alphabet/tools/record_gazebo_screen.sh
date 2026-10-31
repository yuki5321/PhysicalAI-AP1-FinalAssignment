#!/usr/bin/env bash
# Record the container's noVNC desktop (Gazebo window etc.) to MP4 with ffmpeg x11grab.
# Runs INSIDE the container; stops cleanly with Ctrl-C (the MP4 is finalized).
#
#   tools/record_gazebo_screen.sh                      # whole desktop, 30 fps -> videos/screen_<time>.mp4
#   tools/record_gazebo_screen.sh -w                   # click a window (e.g. Gazebo) to record only it
#   tools/record_gazebo_screen.sh -g 1280x720+160+90   # fixed region WxH+X+Y
#   tools/record_gazebo_screen.sh -o /tmp/a.mp4 -r 30 -d :1 -t 120
#
# Options:
#   -o FILE   output (default: <repo>/videos/screen_<YYYYmmdd_HHMMSS>.mp4; *.mp4 is git-ignored)
#   -d DISP   X display (default: $DISPLAY, else the first socket in /tmp/.X11-unix, e.g. :1)
#   -r FPS    frame rate (default 30)
#   -g GEOM   region WxH+X+Y (default: the whole screen)
#   -w        pick a window with the mouse (xwininfo) and record its area
#   -t SEC    stop after SEC seconds (default: until Ctrl-C)
#   -q CRF    x264 quality (default 23; lower = better / bigger)
#   -m        hide the mouse pointer
#
# Finding the display: the noVNC desktop is an Xvnc server inside the container.
# In a terminal opened ON the desktop, `echo $DISPLAY` prints it (usually :1).
# From `docker exec` DISPLAY is not set; `ls /tmp/.X11-unix` lists the sockets
# (X1 -> :1), which this script uses automatically. `xdpyinfo -display :1 | grep dimensions`
# shows the screen size (compose.yaml sets RESOLUTION=1600x900; the noVNC browser
# window may scale it, the recording is always at the real X resolution).
#
# Needs ffmpeg and x11-utils (in docker/Dockerfile; otherwise `sudo apt-get install -y ffmpeg x11-utils`).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
out=""
disp="${DISPLAY:-}"
fps=30
geom=""
pick=0
dur=""
crf=23
mouse=1

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while getopts "o:d:r:g:wt:q:mh" opt; do
  case "$opt" in
    o) out="$OPTARG" ;;
    d) disp="$OPTARG" ;;
    r) fps="$OPTARG" ;;
    g) geom="$OPTARG" ;;
    w) pick=1 ;;
    t) dur="$OPTARG" ;;
    q) crf="$OPTARG" ;;
    m) mouse=0 ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg not found: sudo apt-get install -y ffmpeg" >&2; exit 1; }

if [ -z "$disp" ]; then
  for s in /tmp/.X11-unix/X*; do
    [ -S "$s" ] || continue
    disp=":${s##*/X}"
    break
  done
fi
[ -n "$disp" ] || { echo "no X display found (is the noVNC desktop running?); use -d :1" >&2; exit 1; }
export DISPLAY="$disp"

screen_size() {
  local s=""
  if command -v xdpyinfo >/dev/null; then
    s="$(xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
  fi
  if [ -z "$s" ] && command -v xwininfo >/dev/null; then
    s="$(xwininfo -root 2>/dev/null | awk '/Width:/ {w=$2} /Height:/ {h=$2} END {if (w) print w "x" h}')"
  fi
  echo "$s"   # empty = unknown: let x11grab take the whole screen
}

if [ "$pick" = 1 ]; then
  command -v xwininfo >/dev/null || { echo "xwininfo not found: sudo apt-get install -y x11-utils" >&2; exit 1; }
  echo "Click the window to record (on the noVNC desktop)..."
  info="$(xwininfo)"
  x="$(awk '/Absolute upper-left X:/ {print $4}' <<<"$info")"
  y="$(awk '/Absolute upper-left Y:/ {print $4}' <<<"$info")"
  w="$(awk '/^ *Width:/ {print $2}' <<<"$info")"
  h="$(awk '/^ *Height:/ {print $2}' <<<"$info")"
  geom="${w}x${h}+${x}+${y}"
fi

if [ -n "$geom" ]; then
  size="${geom%%+*}"
  offs="${geom#*+}"
  ox="${offs%%+*}"
  oy="${offs#*+}"
  [ "$ox" -ge 0 ] 2>/dev/null || ox=0
  [ "$oy" -ge 0 ] 2>/dev/null || oy=0
else
  size="$(screen_size)"
  ox=0
  oy=0
fi
# H.264 / yuv420p needs even width and height
if [ -n "$size" ]; then
  w="${size%x*}"; h="${size#*x}"
  size="$((w - w % 2))x$((h - h % 2))"
fi

if [ -z "$out" ]; then
  mkdir -p "$repo/videos"
  out="$repo/videos/screen_$(date +%Y%m%d_%H%M%S).mp4"
fi
mkdir -p "$(dirname "$out")"

args=(-hide_banner -loglevel warning -stats -f x11grab -draw_mouse "$mouse" -framerate "$fps")
[ -n "$size" ] && args+=(-video_size "$size")
args+=(-i "${DISPLAY}.0+${ox},${oy}")
[ -n "$dur" ] && args+=(-t "$dur")
args+=(-vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -preset ultrafast -crf "$crf"
       -pix_fmt yuv420p -movflags +faststart -y "$out")

echo "recording DISPLAY=$DISPLAY ${size:-full screen}+${ox}+${oy} @ ${fps} fps -> $out   (Ctrl-C to stop)"
exec ffmpeg "${args[@]}"
