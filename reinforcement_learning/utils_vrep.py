######################################3
# This is the file containing help functions for controlling vehicle using the vrep
######################################3


import math
import sys
import numpy as np
import vrep

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
