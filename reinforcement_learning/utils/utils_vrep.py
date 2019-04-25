######################################3
# This is the file containing help functions for controlling vehicle using the vrep
######################################3


import math
import sys
import numpy as np
import vrep
from utils.utils_training import getGoalInfo
# Parameters

#########################
# Vehicle Control
#########################

# Set speed of motors
# Input:
#   clientID    : client ID of vrep instance
#   motorHandles: list of integers, denoting motors that you want to change the speed
#   desiredSpd  : single number, speed in km/hr
def setMotorSpeed( clientID, motorHandles, desiredSpd ):
    wheel_radius = 0.63407*0.5      # Wheel radius in metre

    desiredSpd_rps = desiredSpd*(1000/3600)*(1/wheel_radius)   # km/hr into radians per second

    #print("Desired Speed: " + str(desiredSpd) + " km/hr = " + str(desiredSpd_rps) + " radians per seconds = " + str(math.degrees(desiredSpd_rps)) + "degrees per seconds. = " + str(desiredSpd*(1000/3600)) + "m/s" )
    err_code = []
    for mHandle in motorHandles:
        err_code.append( vrep.simxSetJointTargetVelocity(clientID, mHandle, desiredSpd_rps, vrep.simx_opmode_blocking) )
        vrep.simxSetObjectFloatParameter(clientID, mHandle, vrep.sim_shapefloatparam_init_velocity_g, desiredSpd_rps, vrep.simx_opmode_blocking)
    
    
            
    return err_code;
 
# Set Position of motors
# Input:
#   clientID    : client ID of vrep instance
#   motorHandles: 2D list of integers.
#                 [veh1 left, veh1 right]
#                 [veh2 left, veh2 right]...
#   desiredPos  : list of numbers, position in DEGREES
def setMotorPosition( clientID, motorHandles, desiredPos ):
    # Sanity check
    if motorHandles.size != 2*desiredPos.size:
        raise ValueError('input to setMotorPosition is not correct! motorHandles must have 2*size of desiredPos.')

    #print(np.reshape( motorHandles, -1) )
    #print( desiredPos)
    #print( np.radians(desiredPos))
    emptyBuff = bytearray()
    #_ = vrep.simxSetJointTargetPosition(clientID, mHandle, math.radians(desiredPos), vrep.simx_opmode_blocking)
    _, _, _, _, _ = vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'setJointPos_function',np.reshape( motorHandles, -1 ), np.radians(desiredPos),[],emptyBuff,vrep.simx_opmode_blocking)

    return

# Get position of motors
# Input:
#   clientID    : client ID of vrep instance
#   motorHandles: list of integers, denoting motors that you want to change the speed
def getMotorPosition( clientID, motorHandles ):
    motorPos = []
    for mHandle in motorHandles:
        _, pos = vrep.simxGetJointPosition(clientID, mHandle, vrep.simx_opmode_blocking)
        motorPos.append(pos)
    return motorPos

# Get motor orientation
def getMotorOri( clientID, motorHandles ):
    motorOri = np.empty(0)
    for mHandle in motorHandles:
        _, pos = vrep.simxGetJointPosition(clientID, mHandle, vrep.simx_opmode_blocking)
        motorOri = np.append(motorOri, pos)
    return motorOri

# Given one-hot-encoded action array of steering angle, apply it to the vehicle
# Input:
#   action: list of 1/0s. 1 means this action is applied
#   veh_index: index of vehicle
#   options: options containing various info. For this function, we only use ACTION_DIM
#   max_steer: max_steering
#def applySteeringAction(action, veh_index, options, steer_handle, scene_const):
    ## Delta of angle between each action
    #action_delta = (scene_const.max_steer - scene_const.min_steer) / (options.ACTION_DIM-1)
#
    ## Calculate desired angle
    ##action = [0,1,0,0,0]
    #desired_angle = scene_const.max_steer - np.argmax(action) * action_delta
#
    ## Set steering position
    #setMotorPosition(scene_const.clientID, steer_handle[veh_index], desired_angle)
    #
    #return

# Given list of one-hot-encoded action array of steering angle, apply it to the vehicle
# Input:
#   action: 2D array. Each row is list of 1/0s. 1 means this action is applied
#   options: options containing various info. For this function, we only use ACTION_DIM
def applySteeringAction(action_stack, options, handle_dict, scene_const):
    # Delta of angle between each action
    action_delta = (scene_const.max_steer - scene_const.min_steer) / (options.ACTION_DIM-1)
    steer_handle = handle_dict['steer']

    # Calculate desired angle
    #  action = [0,1,0,0,0]
    #  desired_angle is VEH_COUNT x 1 array
    desired_angle = scene_const.max_steer - np.argmax(action_stack,1) * action_delta
   
    # Set steering position
    setMotorPosition(scene_const.clientID, steer_handle, desired_angle)
    
    return

#############################
# Sensors
#############################

# max_distance : normalization factor for sensor readings
def readSensor( sensorHandle, scene_const, op_mode=vrep.simx_opmode_streaming):
    dState      = []
    dDistance   = []
    for sHandle in sensorHandle:
        returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=vrep.simxReadProximitySensor(scene_const.clientID, sHandle, op_mode)
        dState.append(detectionState)
        dDistance.append( np.linalg.norm(detectedPoint) )

    # Set false sensors to max distance
    for i in range(len(dDistance)):
        if dState[i] == 0:
            dDistance[i] = scene_const.max_distance

    # change it into numpy int array
    dState =  np.array(dState)*1
    dDistance = np.array(dDistance)/scene_const.max_distance
    return dState.tolist(), dDistance.tolist()


#############################
#  Getting State of the Vehicle
#############################

# Get current state of the vehicle. It is combination of different information
# Input: veh_index - index of vehicle in vehicle_handle array
#       
# Output: 4 list of float
#  first list    : Sensor distance
#  second list   : Sensor detection state (0-False, 1-True) 
#  third list    : [goal angle, goal distance]
#  fourth list   : vehicle position (x,y)
#  fifth list    : vehicle heading from -1 to 1. 1 if facing left, -1 if facing right, 0 if facing front
def getVehicleState( veh_index, scene_const ):
    # Read sensor
    _, dDistance   = readSensor(sensor_handle[veh_index], scene_const)

    # Read Vehciel Position
    _, vehPos           = vrep.simxGetObjectPosition( scene_const.clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # Read Goal Point Angle & Distance
    gAngle, gDistance   = getGoalPoint( veh_index )

    # Read vehicle heading
    _, vehEuler         = vrep.simxGetObjectOrientation( scene_const.clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # vehEuler - [alpha, beta, gamma] in radians. When vehicle is facing goal point, we have gamma = +90deg = pi/2. Facing right - +0deg, Facing left - +180deg Translate such that it is 0, positive for left, negative for right and within -1 to 1
    vehHeading          = (vehEuler[2]-math.pi*0.5)/(math.pi/2)

    return dDistance, [gAngle, gDistance], vehPos, vehHeading


# Call LUA script to get information
# Returns
#   veh_pos (options.VEH_COUNT x 2)
#   veh_heading (options.VEH_COUNT x 1) [-1,1] -1 if right, 0 if front, 1 if left
#   dDistance (options.VEH_COUNT x SENSOR_COUNT)
#   gInfo (options.VEH_COUNT x 2) [goal angle, goal distance]
def getVehicleStateLUA( handle_list, scene_const):
    emptyBuff = bytearray()

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(scene_const.clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'getVehicleState_function',handle_list,[],[],emptyBuff,vrep.simx_opmode_blocking)

    # Unpack Data
    out_data = np.reshape(retFloats, (-1,3 + 3 + scene_const.sensor_count + 3))       # Reshape such that each row is 18 elements. 3 for pos, 3 for ori, 9 for sensor, 3 for goal pos

    return out_data[:,0:2], ((out_data[:,5]-math.pi*0.5)/scene_const.angle_scale), out_data[:,6:6+scene_const.sensor_count]/scene_const.max_distance, np.transpose( getGoalInfo( out_data[:,0:2], out_data[:,-3:-1], scene_const ) )


##################################
# Getting Handles
##################################


# Get Motor/Sensor Handles
# Input: clientID?
def getMotorHandles( options, scene_const ):
    motor_handle = np.zeros([options.VEH_COUNT,2], dtype=int)
    steer_handle = np.zeros([options.VEH_COUNT,2], dtype=int)

    # Get Motor Handles
    for i in range(0,options.VEH_COUNT):
        _,h1  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_motorLeft" + str(i), vrep.simx_opmode_blocking)
        _,h2  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_motorRight" + str(i), vrep.simx_opmode_blocking)
        _,h3  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_freeAxisRight" + str(i), vrep.simx_opmode_blocking)
        _,h4  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_freeAxisLeft" + str(i), vrep.simx_opmode_blocking)
        _,h5  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_steeringLeft" + str(i), vrep.simx_opmode_blocking)
        _,h6  = vrep.simxGetObjectHandle(scene_const.clientID, "nakedCar_steeringRight" + str(i), vrep.simx_opmode_blocking)

        motor_handle[i][0] = h1
        motor_handle[i][1] = h2

        steer_handle[i][0] = h5
        steer_handle[i][1] = h6

    return motor_handle, steer_handle

def getSensorHandles( options, scene_const ):
    # Get Sensor Handles
    sensor_handle = np.zeros([options.VEH_COUNT,scene_const.sensor_count], dtype=int)

    k = 0
    for v in range(0,options.VEH_COUNT):
        for i in range(0,scene_const.sensor_count):
            _,temp_handle  = vrep.simxGetObjectHandle(scene_const.clientID, "Proximity_sensor" + str(k), vrep.simx_opmode_blocking)
            sensor_handle[v][i] = temp_handle
            k = k+1

    return sensor_handle


##################################
# Handling Simulation
##################################
# Input:
#   step #
#def syncTrigger(scene_const, step_num):
    #emptyBuff = bytearray()
    #_, _, _, _, _ = vrep.simxCallScriptFunction(scene_const.clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'advanceSim_function',[ step_num ], [],[],emptyBuff,vrep.simx_opmode_blocking)
    #
    #return
