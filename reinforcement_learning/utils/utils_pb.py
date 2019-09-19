################
# This file contains functions for pb
###############
import math

import numpy as np
import pybullet as p
from icecream import ic

# import pybullet_data
# import random
# import0os

# Get sensor data
# Output
#   out_distance : 2d array with measurement values 0-1, SIZE : VEH_COUNT x SENSOR_COUNT
#   out_state    : 2d array with measurement state 0 OR 1, SIZE : VEH_COUNT x SENSOR_COUNT
#   out          : concat of above two. SIZE : VEH_COUNT x SENSOR_COUNT*2
def getSensorData( scene_const, options, vehicle_handle ):
    # Initilize variables
    out_distance = np.zeros((options.VEH_COUNT, scene_const.sensor_count)) - 1
    out_state    = np.zeros((options.VEH_COUNT, scene_const.sensor_count)) - 1

    # Loop
    for k in range(0, options.VEH_COUNT):
        # Execute Ray test
        results = p.rayTestBatch(scene_const.rayFrom,scene_const.rayTo, 4, parentObjectUniqueId=vehicle_handle[k], parentLinkIndex=8)

        # Extract hit fraction & detection state
        for j in range(0,scene_const.sensor_count):
            out_distance[k][j] = results[j][2]
            if results[j][0] == -1:
                out_state[k][j] = 1
            else:
                out_state[k][j] = 0

    return np.hstack((out_distance,out_state))

# Get state of the vehicles
# Input
# handle_dict
# Output
#   veh_pos : VEH_COUNT x 2, (x,y)
#   veh_heading : VEH_COUNT x 3, uses euler which is converted from quaternion
#   sensorData : VEH_COUNT x scene_const.sensor_count*2
#   gInfo : VEH_COUNT x 2, [goal_angle, goal_distance]
#       goal_angle : referenced at vehicle heading. left -> -ve, right -> +ve
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
def initQueue(options, sensor_queue, goal_queue, veh_pos_queue, veh_heading_queue, init_veh_pos, init_veh_heading, init_distance, init_gInfo ):
    # For each vehicle
    for i in range(0,options.VEH_COUNT):
        # Copy initial state FRAME_COUNT + 1 times. [0] : oldest info, [-1] : latest info, i.e., current state
        for _ in range(0,options.FRAME_COUNT + 1):
            sensor_queue[i].append(init_distance[i])
            goal_queue[i].append(init_gInfo[i])
            veh_pos_queue[i].append(init_veh_pos[i])
            veh_heading_queue[i].append(init_veh_heading[i])

    return sensor_queue, goal_queue, veh_pos_queue, veh_heading_queue

# Detect whether vehicle is collided
# For now simply say vehicle is collided if distance if smaller than 1.1m. Initial sensor distance to the right starts at 1.2m.
# Input
#   dDistance: array of sensor measurement
#   sensor_count: number of sensors
#   collision_distance: distance to trigger the collision in meters
# Output
#   T/F      : whether collision occured
#   sensor # : Which sensor triggered collision. -1 if no collision
def detectCollision(dDistance, scene_const):
    for i in range(scene_const.sensor_count):
        # collision if distance is short and sensor is closed
        if dDistance[i]*scene_const.sensor_distance < scene_const.collision_distance and dDistance[i + scene_const.sensor_count] == 0:
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
    # if options.VERBOSE == True and options.TESTING == True:
    #     ic('Distance to Goal:', gInfo[1]*scene_const.goal_distance, scene_const.detect_range) 

    if abs(gInfo[1]*scene_const.goal_distance) < scene_const.detect_range:
        return True
    else:
        return False

# Reset the queue by filling the queue with the given initial data
# def resetQueue(options, sensor_queue, goal_queue, dDistance, gInfo, reset_veh_list):
def resetQueue(options, sensor_queue, goal_queue, veh_pos_queue, veh_heading_queue, init_veh_pos, init_veh_heading, init_distance, init_gInfo, reset_veh_list ):
    for v in range(0,options.VEH_COUNT):
        if v in reset_veh_list:
            for _ in range(0,options.FRAME_COUNT + 1):
                # Update queue
                sensor_queue[v].append(init_distance[v])
                sensor_queue[v].popleft()
                goal_queue[v].append(init_gInfo[v])
                goal_queue[v].popleft()
                veh_pos_queue[v].append(init_veh_pos[v])
                veh_pos_queue[v].popleft()
                veh_heading_queue[v].append(init_veh_heading[v])
                veh_heading_queue[v].popleft()

    return veh_pos_queue, veh_heading_queue, sensor_queue, goal_queue

# From sensor and goal queue, get observation.
# observation is a row vector with all frames of information concatentated to each other
# Recall that queue stores 2*FRAME_COUNT of information. 
# first = True: get oldest info
# first = False: get latest info
#
# This work as follows. (Curr frame:3)
#       [frame1, frame2, frame2]
#       old = [frame1,frame2]
#       new = [frame2,frame3]
# Each data is (sensor_count + 2,frame_count). On each frame, last 2 elements are goal point
# Rightmost column is the latest frame
# Output
#   sensor_out : [sensor_count , frame_count] 
#   goal_out   : [2, frame_count] , first row is angle, second row is distance
def getObs( options, scene_const, sensor_queue, goal_queue, old=True):
    if old == True:
        sensor_stack    = np.concatenate(sensor_queue)[0:scene_const.sensor_count*options.FRAME_COUNT]
        goal_stack      = np.concatenate(goal_queue)[0:2*options.FRAME_COUNT]
    else:
        sensor_stack    = np.concatenate(sensor_queue)[scene_const.sensor_count:]
        goal_stack      = np.concatenate(goal_queue)[2:]

    out = np.transpose(np.hstack((sensor_stack.reshape(options.FRAME_COUNT,-1), goal_stack.reshape(options.FRAME_COUNT,-1))))
    # ic(sensor_stack)
    return np.transpose(sensor_stack.reshape(options.FRAME_COUNT,-1)), np.transpose(goal_stack.reshape(options.FRAME_COUNT,-1))
    # return np.concatenate((sensor_stack, goal_stack))

# Update Camera
# Input
#   cam_pos  :  [x,y,z], camera target position
#   cam_dist :  camera distance
# Output
#   cam_pos : new camera position
#   cam_dist : new camerad distance
def controlCamera(cam_pos, cam_dist):
    # get keys
    keys = p.getKeyboardEvents()

    if keys.get(49):  #1
      cam_pos[0] = cam_pos[0] - 1
    if keys.get(50):  #2
      cam_pos[0] = cam_pos[0] + 1
    if keys.get(51):  #3
      cam_pos[1] = cam_pos[1] + 1
    if keys.get(52):  #4
      cam_pos[1] = cam_pos[1] - 1
    if keys.get(53):  #5
      cam_pos = [0,0,0]
      cam_dist = 5
    if keys.get(54):  #6
      cam_dist = cam_dist + 1
    if keys.get(55):  #7
      cam_dist = cam_dist - 1
    p.resetDebugVisualizerCamera( cameraDistance = cam_dist, cameraYaw = 0, cameraPitch = -89, cameraTargetPosition = cam_pos )

    return cam_pos, cam_dist

# Draw Debug Lines
# Input
#   ray_batch_result: result from raytest batch 
#   sensor_data
#   createInit : create initial debug lines
# Output
#   rayIds : 2d array for ray id. each row is ray ids for a single vehicle
def drawDebugLines( options, scene_const, handle_dict, sensor_data = -1, ray_id = -1, createInit = False, ray_width = 2, rayHitColor = [1,0,0], rayMissColor = [0,1,0]):
    # Paremters?
    rayIds=[]                   # 2d array for ray id
    # rayHitColor = [1,0,0]
    # rayMissColor = [0,1,0]
    hokuyo_joint = 8

    # for each vehicle
    for k in range (options.VEH_COUNT):
        rayIds.append([])
        # for each sensor
        for i in range (scene_const.sensor_count):
            # Reference is : y-axis for horizontal, +ve is left.   x-axis for vertical, +ve is up
            # curr_angle = scene_const.sensor_max_angle - i*scene_const.sensor_delta

            # No result just draw
            if createInit == True:
                rayIds[k].append(p.addUserDebugLine(scene_const.rayFrom[i], scene_const.rayTo[i], rayMissColor,parentObjectUniqueId=handle_dict['vehicle'][k], parentLinkIndex=hokuyo_joint ))
            else:
                # Draw
                hit_fraction = sensor_data[k][i]
                if hit_fraction == 1:      # If hit nothing
                    ray_id[k][i] = p.addUserDebugLine(scene_const.rayFrom[i], scene_const.rayTo[i], rayMissColor,parentObjectUniqueId=handle_dict['vehicle'][k], parentLinkIndex=hokuyo_joint, replaceItemUniqueId = ray_id[k][i], lineWidth = ray_width  )
                else:
                    local_hit = np.asarray(scene_const.rayFrom[i]) + hit_fraction*(np.asarray(scene_const.rayTo[i]) - np.asarray(scene_const.rayFrom[i]) )
                    ray_id[k][i] = p.addUserDebugLine(scene_const.rayFrom[i], local_hit, rayHitColor,parentObjectUniqueId=handle_dict['vehicle'][k], parentLinkIndex=hokuyo_joint, replaceItemUniqueId = ray_id[k][i], lineWidth = ray_width )

    
    if ray_id == -1: 
        # If initial creation, return newly created handles
        return rayIds
    else:
        return ray_id
