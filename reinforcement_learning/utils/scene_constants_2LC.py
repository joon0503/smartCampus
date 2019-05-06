#######
# This is class containing all constants relevant to the test scene

import math


class scene_constants:

    # List of constants

    # Maximum distance for sensor readings
    sensor_distance     = 20
    max_distance        = sensor_distance        # DEPRECATED

    # Number of sensors
    sensor_count        = 19

    # Sensor Range in Radians
    #   0 -> front
    #   +\pi/2 -> right
    #   -\pi -> left
    sensor_max_angle    = math.pi/2
    sensor_min_angle    = -math.pi/2

    # Angle (in radians) difference between each radar sensor
    sensor_delta = (sensor_max_angle - sensor_min_angle) / (sensor_count - 1)

    # Contant that scales the angles (i.e., goal point angle)
    angle_scale = math.pi/2
 
    # client ID
    clientID            = -1

    # Maximum steering angle in degrees
    max_steer           = 30
    min_steer           = -1*max_steer

    # Collision occurs if readings are less than this number. in meters
    collision_distance  = 1.3

    # Distance to the goal. Used for normalization
    goal_distance       = 100


    # Simulation Parameters
    dt = 0.025                      # dt of the vrep simulation


    # Test Case related
    obs_w           = 0.4           # obstacle width ratio
    lane_width      = 8             # lane width
    lane_len        = 100           # total len
    turn_len        = 50            # total len after tun
    case_width      = 60           # distance between each case
    veh_init_y      = -45           # initial vehicle y position
