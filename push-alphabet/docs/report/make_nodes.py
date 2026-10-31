# Generates an SVG/HTML node-topic diagram for the report.
import sys
W, H = 1440, 770
o = []
def box(x, y, w, h, title, sub="", color="#2563eb", fill="#eef4ff"):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
    ty = y + h/2 + (-6 if sub else 7)
    o.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="21" font-weight="bold" fill="#111">{title}</text>')
    for i, s in enumerate(sub.split("|") if sub else []):
        o.append(f'<text x="{x+w/2}" y="{ty+24+i*20}" text-anchor="middle" font-size="15" fill="#444">{s}</text>')
def arrow(pts, label="", lx=None, ly=None, color="#333", anchor="middle", dash=False):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    da = ' stroke-dasharray="7,5"' if dash else ""
    o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"{da}/>')
    if label:
        o.append(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="16" font-family="monospace" fill="{color}">{label}</text>')
def region(x, y, w, h, label, color):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="9,6"/>')
    o.append(f'<text x="{x+12}" y="{y+22}" font-size="16" fill="{color}" font-weight="bold">{label}</text>')

region(300, 8, 1130, 752, "Docker コンテナ（ROS 2 Humble）", "#64748b")
region(8, 520, 280, 240, "Windows ホスト", "#64748b")
# Gazebo
box(330, 40, 1050, 90, "Gazebo Classic 11 の世界", "台・壁・文字ブロック・押し棒（オレンジの円柱 point_pusher）・真上カメラ 640×640", "#475569", "#f1f5f9")
# row 2
box(330, 230, 250, 90, "letter_detector", "pa_perception|classic / cnn / hybrid", "#16a34a", "#ecfdf3")
box(700, 322, 260, 128, "planner_node", "pa_planner|状態機械 + PointPushPlanner|（2D 押しモデルで先読み）", "#d97706", "#fff8e6")
box(1060, 230, 290, 90, "point_push_executor", "pa_gazebo|降ろす → 直線で押す → 持ち上げる", "#ea580c", "#fff1e8")
box(1080, 450, 250, 90, "scene_manager", "pa_gazebo|文字の出し入れ・雪の演出", "#475569", "#f8fafc")
box(1080, 630, 250, 90, "hand_actor", "pa_gazebo|シミュ内の手を動かす", "#db2777", "#fdf0f6")
box(330, 560, 250, 80, "rosbridge", "rosbridge_websocket :9090", "#475569", "#f8fafc")
box(24, 560, 250, 130, "hand_bridge.py", "Web カメラ|→ MediaPipe Hands|→ 掌の位置（台の座標）", "#db2777", "#fdf0f6")
# arrows
arrow([(455,130),(455,228)], "/overhead_camera/image_raw", 465, 185, anchor="start")
arrow([(580,275),(640,275),(640,365),(698,365)], "/letters", 588, 262, anchor="start")
arrow([(960,345),(1020,345),(1020,262),(1058,262)], "/point_push_cmd", 1012, 254, anchor="end")
arrow([(1060,300),(1045,300),(1045,372),(962,372)], "/point_push_done", 968, 392, anchor="start")
arrow([(1205,230),(1205,132)], "/set_entity_state（押し棒を動かす）", 1195, 185, anchor="end")
arrow([(962,425),(1040,425),(1040,480),(1078,480)], "/letter_done", 968, 418, anchor="start", color="#475569")
arrow([(1080,525),(900,525),(900,452)], "/active_letter", 1070, 517, anchor="end", color="#475569")
arrow([(1330,495),(1370,495),(1370,132)], "", color="#475569")
arrow([(1330,675),(1392,675),(1392,132)], "", color="#db2777")
o.append('<text x="1416" y="330" font-size="15" fill="#475569" transform="rotate(-90 1416 330)" text-anchor="middle">/set_entity_state（文字・手を動かす）</text>')
arrow([(274,600),(328,600)], "", color="#db2777")
o.append('<text x="301" y="590" font-size="14" fill="#db2777" text-anchor="middle">ws</text>')
arrow([(580,600),(800,600),(800,452)], "/hand/state", 810, 560, anchor="start", color="#db2777")
arrow([(690,600),(690,700),(1078,700)], "/hand/state", 700, 690, anchor="start", color="#db2777")
# real-robot notes
o.append('<text x="455" y="345" font-size="14" fill="#2563eb" text-anchor="middle"></text>')
o.append('<text x="1205" y="345" font-size="15" fill="#2563eb" text-anchor="middle">実機: XY ステージ / アームのドライバ</text>')
o.append('<text x="845" y="160" font-size="15" fill="#2563eb" text-anchor="middle">実機: Gazebo → 台・XY ステージ＋押し棒・usb_cam に置き換え（トピックは同じ）</text>')


svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="IPAGothic, sans-serif">
<defs><marker id="ah" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="context-stroke"/></marker></defs>
<rect width="{W}" height="{H}" fill="white"/>
{chr(10).join(o)}
</svg>'''
open(sys.argv[1], "w").write(f'<!doctype html><meta charset="utf-8"><style>body{{margin:0}}</style>{svg}')
