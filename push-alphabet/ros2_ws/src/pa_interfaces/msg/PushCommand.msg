# One paddle swing: rotate `side` from rest to `angle_deg`, then back to rest.
# Maps 1:1 to the Arduino serial command on the real robot.
uint8 SOUTH=0
uint8 EAST=1
uint8 NORTH=2
uint8 WEST=3
std_msgs/Header header
uint32 id            # echoed on /push_done when the swing has finished
uint8 side           # index into scene.yaml pushers.list
float32 angle_deg    # 0 .. pushers.max_angle
