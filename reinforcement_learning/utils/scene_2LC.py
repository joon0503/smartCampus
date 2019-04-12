##############3
# This file is functions related to each scene.
#
#
# Scenario : 2 lane change and get to the goal point

import vrep
import random
import math
from utils.utils_vrep import * 





# Initialize to original scene
# Input:
#   scene_const
#   options
#   handle_dict : dictionary of handle list
def initScene( scene_const, options, veh_index_list, handle_dict, randomize = False):

    vehicle_handle = handle_dict['vehicle']
    steer_handle   = handle_dict['steer']
    motor_handle   = handle_dict['motor']
    obs_handle     = handle_dict['obstacle']
    sensor_handle  = handle_dict['sensor']

    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( scene_const.clientID, vehicle_handle[veh_index], vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
        # Reset position of vehicle. Randomize x-position if enabled
        if randomize == False:
            err_code = vrep.simxSetObjectPosition(scene_const.clientID,vehicle_handle[veh_index],-1,[veh_index*20-3 + 0,0,0.2],vrep.simx_opmode_blocking)
        else:
            x_pos = random.uniform(-1,-3)
            err_code = vrep.simxSetObjectPosition(scene_const.clientID,vehicle_handle[veh_index],-1,[veh_index*20 + x_pos,0,0.2],vrep.simx_opmode_blocking)

        # Reset Orientation of vehicle
        err_code = vrep.simxSetObjectOrientation(scene_const.clientID,vehicle_handle[veh_index],-1,[0,0,math.radians(90)],vrep.simx_opmode_blocking)

        # Reset position of motors & steering
        setMotorPosition(scene_const.clientID, steer_handle[veh_index], 0)

    vrep.simxSynchronousTrigger(scene_const.clientID);                              # Step one simulation while dynamics disabled to move object

    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( scene_const.clientID, vehicle_handle[veh_index], 0 , vrep.simx_opmode_blocking   )      # enable dynamics

        # Reset motor speed
        setMotorSpeed(scene_const.clientID, motor_handle[veh_index], options.INIT_SPD)

        # Set initial speed of vehicle
        vrep.simxSetObjectFloatParameter(scene_const.clientID, vehicle_handle[veh_index], vrep.sim_shapefloatparam_init_velocity_y, options.INIT_SPD/3.6, vrep.simx_opmode_blocking)
       
        # Read sensor
        dState, dDistance = readSensor(sensor_handle[veh_index], scene_const, vrep.simx_opmode_buffer)         # try it once for initialization

        # Reset position of obstacle
        if randomize == True:
            if random.random() > 0.5:
                err_code = vrep.simxSetObjectPosition(scene_const.clientID,obs_handle[veh_index],-1,[veh_index*20 + -3.3,30,1.1],vrep.simx_opmode_blocking)
            else:
                err_code = vrep.simxSetObjectPosition(scene_const.clientID,obs_handle[veh_index],-1,[veh_index*20 + 1.675,30,1.1],vrep.simx_opmode_blocking)
        
        # Reset position of dummy    
        if randomize == False:
            pass
        else:
            pass
    #        x_pos = random.uniform(-1,-7.25)
    #        err_code = vrep.simxSetObjectPosition(scene_const.clientID,dummy_handle,-1,[x_pos,60,0.2],vrep.simx_opmode_blocking)
