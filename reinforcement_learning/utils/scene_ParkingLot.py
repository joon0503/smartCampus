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

    # If reset list is empty, just return
    if len(veh_index_list) == 0:
        return


    vehicle_handle = handle_dict['vehicle']
    steer_handle   = handle_dict['steer']
    motor_handle   = handle_dict['motor']
    sensor_handle  = handle_dict['sensor']

    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( scene_const.clientID, vehicle_handle[veh_index], vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
        # Reset position of vehicle. Randomize x-position if enabled
        err_code = vrep.simxSetObjectPosition(scene_const.clientID,vehicle_handle[veh_index],-1,[70,-10,0.2],vrep.simx_opmode_blocking)

        # Reset Orientation of vehicle
        err_code = vrep.simxSetObjectOrientation(scene_const.clientID,vehicle_handle[veh_index],-1,[0,0,0],vrep.simx_opmode_blocking)

    # Reset position of motors & steering
    setMotorPosition(scene_const.clientID, handle_dict['steer'], np.zeros(options.VEH_COUNT))

    vrep.simxSynchronousTrigger(scene_const.clientID);                              # Step one simulation while dynamics disabled to move object

    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( scene_const.clientID, vehicle_handle[veh_index], 0 , vrep.simx_opmode_blocking   )      # enable dynamics

        # Reset motor speed
        setMotorSpeed(scene_const.clientID, motor_handle[veh_index], options.INIT_SPD)

        # Set initial speed of vehicle
        vrep.simxSetObjectFloatParameter(scene_const.clientID, vehicle_handle[veh_index], vrep.sim_shapefloatparam_init_velocity_x, options.INIT_SPD/3.6, vrep.simx_opmode_blocking)
       
        # Read sensor
        dState, dDistance = readSensor(sensor_handle[veh_index], scene_const, vrep.simx_opmode_buffer)         # try it once for initialization

    return




