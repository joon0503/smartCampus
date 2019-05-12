import pybullet as p
import math
import numpy as np


######################################
# Helper Functions for PyBullet
######################################



# get vehicle state






# Get goal info
# Input
#   car_id
#   goal_id
# Output
#   goal_angle
#   goal_distance
def getGoalInfo( car_id, goal_id, scene_const):
    vehPos, _   = p.getBasePositionAndOrientation( car )
    goalPos, _ = p.getBasePositionAndOrientation( goal_id )

    # Calculate the distance
    delta_distance  = np.array(goalPos[0:1]) - np.array(vehPos[0:1])              # delta x, delta y
    goal_distance   = np.linalg.norm(delta_distance)

    # calculate angle
    goal_angle = math.atan( delta_distance[0]/delta_distance[1] )       # delta x / delta y
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
    #if abs(gInfo[1]*scene_const.goal_distance - 2.075) < 1.0 and abs(currHeading*90)<5: 
    if abs(gInfo[1]*scene_const.goal_distance - 2.075) < 1.0: 
        return True
    else:
        return False
