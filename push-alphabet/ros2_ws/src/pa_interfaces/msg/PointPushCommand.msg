# One straight push with the point pusher (Push-T style): lower the pusher at
# (x0, y0), move it in a straight line to (x1, y1), then lift it and park.
# World frame of scene.yaml [m]. Echoed on /point_push_done when finished.
std_msgs/Header header
uint32 id
float64 x0
float64 y0
float64 x1
float64 y1
float32 speed        # [m/s]; 0 = scene.yaml point_pusher.speed
