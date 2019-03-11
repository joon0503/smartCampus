#######
# This is class containing all constants relevant to the test scene



class scene_constants:

    # List of constants

    # Maximum distance for sensor readings
    max_distance        = 20

    # Number of sensors
    sensor_count        = 9

    # client ID
    clientID            = -1

    # Maximum steering angle in degrees
    mas_steer           = 15
    min_steer           = -15

    # Collision occurs if readings are less than this number. in meters
    collision_distance  = 1.3

    # Distance to the goal. Used for normalization
    goal_distance       = 60

