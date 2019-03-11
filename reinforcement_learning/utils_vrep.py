######################################3
# This is the file containing help functions for controlling vehicle using the vrep
######################################3


import math
import sys
import numpy as np
import vrep
from utils_training import getGoalInfo
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
#   motorHandles: list of integers, denoting motors that you want to change the speed
#   desiredPos  : single number, position in DEGREES
#               : positive angle means turns LEFT
def setMotorPosition( clientID, motorHandles, desiredPos ):
    for mHandle in motorHandles:
        _ = vrep.simxSetJointTargetPosition(clientID, mHandle, math.radians(desiredPos), vrep.simx_opmode_blocking)

    return;

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
def applySteeringAction(action, veh_index, options, clientID, steer_handle, max_steer = 15):
    # Define maximum/minimum steering in degrees
    min_steer = -1*max_steer

    # Delta of angle between each action
    action_delta = (max_steer - min_steer) / (options.ACTION_DIM-1)

    # Calculate desired angle
    #action = [0,1,0,0,0]
    desired_angle = max_steer - np.argmax(action) * action_delta

    # Set steering position
    setMotorPosition(clientID, steer_handle[veh_index], desired_angle)
    
    return


#############################
# Sensors
#############################

# max_distance : normalization factor for sensor readings
def readSensor( clientID, sensorHandle, op_mode=vrep.simx_opmode_streaming, max_distance = 20 ):
    dState      = []
    dDistance   = []
    for sHandle in sensorHandle:
        returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID, sHandle, op_mode)
        dState.append(detectionState)
        dDistance.append( np.linalg.norm(detectedPoint) )

    # Set false sensors to max distance
    for i in range(len(dDistance)):
        if dState[i] == 0:
            dDistance[i] = max_distance

    # change it into numpy int array
    dState =  np.array(dState)*1
    dDistance = np.array(dDistance)/max_distance
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
def getVehicleState( veh_index ):
    # Read sensor
    _, dDistance   = readSensor(clientID, sensor_handle[veh_index])

    # Read Vehciel Position
    _, vehPos           = vrep.simxGetObjectPosition( clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # Read Goal Point Angle & Distance
    gAngle, gDistance   = getGoalPoint( veh_index )

    # Read vehicle heading
    _, vehEuler         = vrep.simxGetObjectOrientation( clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # vehEuler - [alpha, beta, gamma] in radians. When vehicle is facing goal point, we have gamma = +90deg = pi/2. Facing right - +0deg, Facing left - +180deg Translate such that it is 0, positive for left, negative for right and within -1 to 1
    vehHeading          = (vehEuler[2]-math.pi*0.5)/(math.pi/2)

    return dDistance, [gAngle, gDistance], vehPos, vehHeading


# Call LUA script to get information
# Returns
#   veh_pos (options.VEH_COUNT x 2)
#   veh_heading (options.VEH_COUNT x 1) [-1,1] -1 if right, 0 if front, 1 if left
#   dDistance (options.VEH_COUNT x SENSOR_COUNT)
#   gInfo (options.VEH_COUNT x 2)
def getVehicleStateLUA( clientID, handle_list, sensor_count = 9, max_distance = 20):
    emptyBuff = bytearray()

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'getVehicleState_function',handle_list,[],[],emptyBuff,vrep.simx_opmode_blocking)

    # Unpack Data
    out_data = np.reshape(retFloats, (-1,18))       # Reshape such that each row is 18 elements. 3 for pos, 3 for ori, 9 for sensor, 3 for goal pos

    return out_data[:,0:2], ((out_data[:,5]-math.pi*0.5)/(math.pi/2)), out_data[:,6:6+sensor_count]/max_distance, np.transpose( getGoalInfo( out_data[:,0:2], out_data[:,15:17] ) )


##################################
# Getting Handles
##################################


# Get Motor/Sensor Handles
# Input: clientID?
def getMotorHandles( clientID, options ):
    motor_handle = np.zeros([options.VEH_COUNT,2], dtype=int)
    steer_handle = np.zeros([options.VEH_COUNT,2], dtype=int)

    # Get Motor Handles
    for i in range(0,options.VEH_COUNT):
        _,h1  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorLeft" + str(i), vrep.simx_opmode_blocking)
        _,h2  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorRight" + str(i), vrep.simx_opmode_blocking)
        _,h3  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisRight" + str(i), vrep.simx_opmode_blocking)
        _,h4  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisLeft" + str(i), vrep.simx_opmode_blocking)
        _,h5  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringLeft" + str(i), vrep.simx_opmode_blocking)
        _,h6  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringRight" + str(i), vrep.simx_opmode_blocking)

        motor_handle[i][0] = h1
        motor_handle[i][1] = h2

        steer_handle[i][0] = h5
        steer_handle[i][1] = h6

    return motor_handle, steer_handle

def getSensorHandles( clientID, options, sensor_count = 9 ):
    # Get Sensor Handles
    sensor_handle = np.zeros([options.VEH_COUNT,sensor_count], dtype=int)

    k = 0
    for v in range(0,options.VEH_COUNT):
        for i in range(0,sensor_count):
            _,temp_handle  = vrep.simxGetObjectHandle(clientID, "Proximity_sensor" + str(k), vrep.simx_opmode_blocking)
            sensor_handle[v][i] = temp_handle
            k = k+1

    return sensor_handle
