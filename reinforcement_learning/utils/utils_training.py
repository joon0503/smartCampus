import numpy as np
import vrep
import math
import sys


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
        if dDistance[i]*scene_const.max_distance < scene_const.collision_distance:
            return True, i
    
    return False, -1    

# Get goal point vector. Returns angle and distance
# Input
# goal_distance_base : normalizing factor for goal distance
# Output
# goal_angle: angle to the goal point in radians / pi
#             Positive angle mean goal point is on the right of vehicle, negative mean goal point is on the left
# goal_distance: distance to the goal point / MAX_DISTANCE
def getGoalPoint( veh_index, scene_const ):
    _, vehPos = vrep.simxGetObjectPosition( scene_const.clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            
    _, dummyPos = vrep.simxGetObjectPosition( scene_const.clientID, dummy_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # Calculate the distance
    delta_distance = np.array(dummyPos) - np.array(vehPos)              # delta x, delta y
    goal_distance = np.linalg.norm(delta_distance)

    # calculate angle
    goal_angle = math.atan( delta_distance[0]/delta_distance[1] )       # delta x / delta y
    goal_angle = goal_angle / math.pi                               # result in -1 to 1

    return goal_angle, goal_distance / scene_const.goal_distance

# veh_pos_info: (options.VEH_COUNT x 2)
def getGoalInfo( veh_pos_info, goal_pos_info, scene_const ):
    # Calculate the distance
    delta_distance = goal_pos_info - veh_pos_info              # delta x, delta y
    #print(goal_pos_info)
    #print(veh_pos_info)
    #print(delta_distance)
    goal_distance = np.linalg.norm(delta_distance, axis=1)

    # calculate angle
    goal_angle = np.arctan( delta_distance[:,0]/delta_distance[:,1] )       # delta x / delta y
    goal_angle = goal_angle / math.pi                               # result in -1 to 1

    return goal_angle, goal_distance / scene_const.goal_distance



# Detect whether vehicle reached goal point.
# Input
#   vehPos - [x,y,z] - center of vehicle. To get tip, add 2.075m
#   gInfo  - [angle,distance]
# Output
#   True/False
def detectReachedGoal(vehPos, gInfo, currHeading, scene_const ):
    # Distance less than 0.5m, angle less than 10 degrees
    if abs(gInfo[1]*scene_const.goal_distance - 2.075) < 1.0 and abs(currHeading*90)<5: 
        return True
    else:
        return False

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
        for q_reset in range(0,options.FRAME_COUNT*2):
            sensor_queue[i].append(dDistance[i])
            goal_queue[i].append(gInfo[i])


# Reset the queue by filling the queue with the given initial data
def resetQueue(options, sensor_queue, goal_queue, dDistance, gInfo):
    for v in range(0,len(sensor_queue)):
        for q in range(0,options.FRAME_COUNT*2):
            # Update queue
            sensor_queue[v].append(dDistance[v])
            sensor_queue[v].popleft()
            goal_queue[v].append(gInfo[v])
            goal_queue[v].popleft()

    return sensor_queue, goal_queue

    



