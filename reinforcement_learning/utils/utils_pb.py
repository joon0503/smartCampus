################
# This file contains functions for pb
###############
import numpy as np
import pybullet as p
from icecream import ic
import math
# import pybullet_data
# import random
# import0os

# Get sensor data
# Output
#   out : 2d array with measurement values 0-1
def getSensorData( scene_const, options, vehicle_handle ):
    # Initilize variables
    out = np.zeros((options.VEH_COUNT, scene_const.sensor_count)) - 1

    # Loop
    for k in range(0, options.VEH_COUNT):
        # Execute Ray test
        results = p.rayTestBatch(scene_const.rayFrom,scene_const.rayTo, 4, parentObjectUniqueId=vehicle_handle[k], parentLinkIndex=8)

        # Extract hit fraction
        for j in range(0,scene_const.sensor_count):
            out[k][j] = results[j][2]

    return out 

# Get state of the vehicles
# Input
# handle_dict
# Output
#   veh_pos : VEH_COUNT x 2, (x,y)
#   veh_heading : VEH_COUNT x 4, uses quaternion
#   sensorData : VEH_COUNT x scene_const.sensor_count
#   gInfo : VEH_COUNT x 2, [goal_angle, goal_distance]
def getVehicleState( scene_const, options, handle_dict ):
    veh_pos     = np.zeros((options.VEH_COUNT,2))
    veh_heading = np.zeros((options.VEH_COUNT,3))
    gInfo       = np.zeros((options.VEH_COUNT,2))

    # Get Handles
    vehicle_handle  = handle_dict['vehicle']
    goal_handle     = handle_dict['dummy']


    for k in range(options.VEH_COUNT):
        # Get vehicle position & orientation
        temp, temp_heading = p.getBasePositionAndOrientation( vehicle_handle[k] )
        veh_pos[k] = temp[0:2]
        veh_heading[k] = p.getEulerFromQuaternion( temp_heading )       # It seems like veh_heading[k][2] is the heading of vehicle in radians with 0 being heading to east, and heading north is pi/2.

        # To compute gInfo, get goal position
        g_pos, _ = p.getBasePositionAndOrientation( goal_handle[k] )

        # Calculate the distance
        # ic(g_pos[0:2],veh_pos[k])
        delta_distance = np.array(g_pos[0:2]) - np.array(veh_pos[k])              # delta x, delta y
        gInfo[k][1]  = np.linalg.norm(delta_distance) / scene_const.goal_distance

        # calculate angle. 90deg + angle (assuming heading north) - veh_heading
        gInfo[k][0] = math.atan( abs(delta_distance[0])/abs(delta_distance[1]) )      # delta x / delta y, and scale by pi 1(left) to -1 (right)
        if delta_distance[0] < 0 :
            # Goal is left of the vehicle, then -1
            gInfo[k][0] = gInfo[k][0]*-1

        # Scale with heading
        gInfo[k][0] = -1*(math.pi*0.5 - gInfo[k][0] - veh_heading[k][2] ) / (math.pi/2)
    return veh_pos, veh_heading, getSensorData( scene_const, options, vehicle_handle ), gInfo

# Initialize the array of queue
# Input
#   options
#   sensor_queue : empty array
#   goal_queue   : empty array
#   dDistance    : array (for each vehicle) of sensor measurements
#   gInfo        : array (for each vehicle) of goal measurements
def initQueue(options, sensor_queue, goal_queue, dDistance, gInfo):
    for i in range(0,options.VEH_COUNT):
        # Copy initial state FRAME_COUNT*2 times. First FRAME_COUNT will store state of previous, and last FRAME_COUNT store state of current
        for _ in range(0,options.FRAME_COUNT*2):
            sensor_queue[i].append(dDistance[i])
            goal_queue[i].append(gInfo[i])

    return sensor_queue, goal_queue

# Detect whether vehicle is collided
# For now simply say vehicle is collided if distance if smaller than 1.1m. Initial sensor distance to the right starts at 1.2m.
# Input
#   dDistance: array of sensor measurement
#   sensor_count: number of sensors
#   collision_distance: distance to trigger the collision in meters
# Output
#   T/F      : whether collision occured
#   sensor # : Which sensor triggered collision. -1 if no collision
def detectCollision(dDistance, scene_const ):
    for i in range(scene_const.sensor_count):
        if dDistance[i]*scene_const.sensor_distance < scene_const.collision_distance:
            return True, i
    
    return False, -1    

# Detect whether vehicle reached goal point.
# Input
#   vehPos - [x,y,z] - center of vehicle. To get tip, add 2.075m
#   gInfo  - [angle,distance]
# Output
#   True/False
def detectReachedGoal(vehPos, gInfo, currHeading, scene_const ):
    # Distance less than 0.5m, angle less than 10 degrees
    #if abs(gInfo[1]*scene_const.goal_distance - 2.075) < 1.0 and abs(currHeading*90)<5: 
    if abs(gInfo[1]*scene_const.goal_distance) < scene_const.detect_range: 
        return True
    else:
        return False

# Reset the queue by filling the queue with the given initial data
def resetQueue(options, sensor_queue, goal_queue, dDistance, gInfo, reset_veh_list):
    for v in range(0,options.VEH_COUNT):
        if v in reset_veh_list:
            for _ in range(0,options.FRAME_COUNT*2):
                # Update queue
                sensor_queue[v].append(dDistance[v])
                sensor_queue[v].popleft()
                goal_queue[v].append(gInfo[v])
                goal_queue[v].popleft()

    return sensor_queue, goal_queue
