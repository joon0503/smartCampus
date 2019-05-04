########
# This file generate the scene file for Reinforcedment learning
# You must open vrep, set simulation option such that it keeps newly added objects.
# Also, it requires dyros_vehicle to be present
# Then run this script.

# This particular script generates lane-change scenario with obstacle.
#######

import time



###############
# Helper
###############

# Create Object
# objSize: size of object, triple
# objPos : position of object, triple
# objName: name of object, string
def createObject(objSize, objPos, objName, parentHandle=-1):
    # First remove object if one exists with same name
    retCode, retHandle = vrep.simxGetObjectHandle(clientID, objName, vrep.simx_opmode_blocking)

    # retCode == 0 means function executed, i.e., there is already a another object
    if retCode == 0:
        print(objName + " already exists. It will be removed first.")
        vrep.simxRemoveObject(clientID, retHandle, vrep.simx_opmode_blocking)

    emptyBuff = bytearray()
    inputFloats = objSize + objPos #concatenation

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'createCuboid_function',[parentHandle],inputFloats,[objName],emptyBuff,vrep.simx_opmode_blocking)
    if res==vrep.simx_return_ok:
        print ( objName + ' created!')
    else:
        print ('Creating object failed.')

    # return handle
    return retInts

# Create Dummy
# objPos : position of object, triple
# objName: name of object, string
# parentHandle: handle of parent. -1 if no parent
def createDummy(objPos, objName, parentHandle=-1):
    # First remove object if one exists with same name
    retCode, retHandle = vrep.simxGetObjectHandle(clientID, objName, vrep.simx_opmode_blocking)

    # retCode == 0 means function executed, i.e., there is already a another object
    if retCode == 0:
        print(objName + " already exists. It will be removed first.")
        vrep.simxRemoveObject(clientID, retHandle, vrep.simx_opmode_blocking)

    emptyBuff = bytearray()
    inputFloats = objPos #concatenation

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'createDummy_function',[parentHandle],inputFloats,[objName],emptyBuff,vrep.simx_opmode_blocking)
    if res==vrep.simx_return_ok:
        print ( objName + ' created!')
    else:
        print ('Creating dummy failed.')

    # return handle
    return retInts

def errorExit():
    # Now close the connection to V-REP:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)
    sys.exit()
    return

##############################
# Helper Generate a Test Scenario
##############################

# Generate straight case
# input
def genStraight( lane_width, lane_len, obs_w, x_pos, i, getInit = False ):
    # Case Parameters
    wall_h      = 2.5

    if getInit == True:
        return -1*lane_len*0.5 + 5

    
    # Floor
    createObject([lane_width, lane_len, 0.1],[x_pos, 0,-0.1],'floor' + str(i), scene_handle)               

    # Walls
    createObject([0.1, lane_len, wall_h],[x_pos - lane_width*0.5, 0, wall_h*0.5],'wallLeft' + str(i), scene_handle)                     # left
    createObject([0.1, lane_len, wall_h],[x_pos + lane_width*0.5, 0, wall_h*0.5],'wallRight' + str(i), scene_handle)                    # right
    createObject([lane_width, 0.2 , wall_h],[x_pos,lane_len*0.5, wall_h*0.5],'wallEnd' + str(i), scene_handle)                          # End
    createObject([lane_width, 0.2 , wall_h],[x_pos,-1*lane_len*0.5, wall_h*0.5],'wallBack' + str(i), scene_handle)                          # back

    # Goal Point
    createDummy([x_pos, 0.5*lane_len - 5, 0.3], 'GoalPoint' + str(i), parentHandle=scene_handle)                        # Create Goal Point

    # Obstacles
    createObject([lane_width*obs_w, 5, 2], [x_pos + lane_width*0.25, 0, 1.1], 'obstacle' + str(i) + str(1), scene_handle)                # create obstacle



    return

# Generate a case of T intersection
#   lane_width : width of lane
#   lane_len   : length of lane
#   obs_w : width of obs w.r.t. lane width. Ranges 0 to 1
#   x_pos : x position of the case
#   i : index for naming
#   turn_len : length of Tee
#   goal direction : 0/1/2 - direction of the goal point
#               0 : left
#               1 : straight
#               2 : right
#   openWall - T/F
#       T : does not block
#       F : block opposite direction
#   getInit : True - return initial vehicle position / False - actually generate case
# output
#   veh_y : if true, just returns the initial y position of the test case
def genTee( lane_width, turn_len, lane_len, obs_w, x_pos, i, getInit = False, direction = -1, openWall = True):
    ##############################
    # Case Parameters
    ##############################
    wall_h      = 2.5

    # Returns init pos
    if getInit == True:
        return -1*lane_len*0.5 + 5

    ##############################
    # Floor
    ##############################
    # Staight
    createObject([lane_width, lane_len, 0.1],[x_pos, 0,-0.1],'floor' + str(i), scene_handle)                                    

    # Tee
    createObject([turn_len, lane_width, 0.1],[x_pos, 0.5*(lane_len - lane_width),-0.1],'floor' + str(i) + str(2), scene_handle)          

    ##############################
    # Walls
    ##############################
    if openWall == True:
        delta_left      = -1*lane_width
        delta_right     = -1*lane_width
    else:
        if direction == 0:
            delta_left      = -1*lane_width
            delta_right     = 0
        elif direction == 1:
            delta_left      = 0
            delta_right     = 0
        elif direction == 2:
            delta_left      = 0
            delta_right     = -1*lane_width
        else:
            raise ValueError('Unspecified direction argument!')

    # Left
    createObject([0.1, lane_len, wall_h],[x_pos - lane_width*0.5, delta_left, wall_h*0.5],'wallLeft' + str(i), scene_handle)                   

    # Right
    createObject([0.1, lane_len, wall_h],[x_pos + lane_width*0.5, delta_right, wall_h*0.5],'wallRight' + str(i), scene_handle)               

    # Up
    createObject([turn_len, 0.2 , wall_h],[x_pos,lane_len*0.5, wall_h*0.5],'wallUp' + str(i), scene_handle)                       

    # Rest
    rest_len = 0.5*(turn_len - lane_width)
    createObject([ rest_len, 0.2 , wall_h],[x_pos - lane_width*0.5 - rest_len*0.5, lane_len*0.5 - lane_width, wall_h*0.5],'wallRest' + str(i) + str(1), scene_handle)                   
    createObject([ rest_len, 0.2 , wall_h],[x_pos + lane_width*0.5 + rest_len*0.5, lane_len*0.5 - lane_width, wall_h*0.5],'wallRest' + str(i) + str(2), scene_handle)                   


    # Walls at the end
    #createObject([lane_width, 0.2 , wall_h],[x_pos,-1*lane_len*0.5, wall_h*0.5],'wallEnd_left' + str(i), scene_handle)                        
    createObject([0.2, lane_width , wall_h],[x_pos + lane_width*0.5 + rest_len, 0.5*lane_len - 0.5*lane_width, wall_h*0.5],'wallEnd_right' + str(i), scene_handle)                        
    createObject([0.2, lane_width , wall_h],[x_pos - lane_width*0.5 - rest_len, 0.5*lane_len - 0.5*lane_width, wall_h*0.5],'wallEnd_left' + str(i), scene_handle)                        

    ##############################
    # Goal Point
    ##############################
    if direction == 0:
        createDummy([x_pos - turn_len*0.5 + 3, 0.5*(lane_len - lane_width), 0.3], 'GoalPoint' + str(i), parentHandle=scene_handle)                        # Create Goal Point
    elif direction == 1:
        createDummy([x_pos, 0.5*(lane_len - lane_width), 0.3], 'GoalPoint' + str(i), parentHandle=scene_handle)                        # Create Goal Point
    elif direction == 2:
        createDummy([x_pos + turn_len*0.5 - 3, 0.5*(lane_len - lane_width), 0.3], 'GoalPoint' + str(i), parentHandle=scene_handle)                        # Create Goal Point
    else:
        raise ValueError('Unspecified direction argument!')

    ##############################
    # Obstacles
    ##############################
    createObject([lane_width*obs_w, 5, 2], [x_pos + 0.5*(1-obs_w)*lane_width, 0, 1.1], 'obstacle' + str(i) + str(1), scene_handle)                # create obstacle

    return

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import sys
import ctypes
import math
import numpy as np

#################
#MAIN
#################
print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    print ('Connected to remote API server')
else:
    print ('Failed connecting to remote API server')
    sys.exit()


# Parameters 
COPY_NUM = 12
case_width = 100                 # distance between each test case

# Handle
err_code,dyros_handle = vrep.simxGetObjectHandle(clientID,"dyros_vehicle", vrep.simx_opmode_blocking) 

for i in range(0,COPY_NUM):
    # create vehicle
    err_code1, vehicle_handle = vrep.simxCopyPasteObjects(clientID,[dyros_handle], vrep.simx_opmode_blocking)
    vehicle_handle = vehicle_handle[0]

    #######################
    # Create Sensors
    #######################
    SENSOR_COUNT = 19
    RAD_DT = math.pi/(SENSOR_COUNT-1)

    sensor_handle_array = [0]
    err_code,sensor_handle_array[0] = vrep.simxGetObjectHandle(clientID,"Proximity_sensor", vrep.simx_opmode_blocking)
    if err_code != 0:
       print("ERROR: No sensor found.")
       errorExit() 

    for s in range(0,SENSOR_COUNT):
        print(str(s))
        err_code1, sensor_handle = vrep.simxCopyPasteObjects(clientID,[sensor_handle_array[0]], vrep.simx_opmode_blocking)
        vrep.simxSetObjectParent(clientID,sensor_handle[0],vehicle_handle,False,vrep.simx_opmode_blocking)
        ret_code = vrep.simxSetObjectPosition(clientID,sensor_handle[0],vehicle_handle,[1.5,0,0.1],vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(clientID,sensor_handle[0],vehicle_handle,[-math.pi/2,RAD_DT*s,0],vrep.simx_opmode_oneshot)
        sensor_handle_array = np.append(sensor_handle_array,sensor_handle)   

    #######################
    # Paremters
    #######################
    lane_width = 8
    lane_len   = 100
    obs_w      = 0.5
    turn_len   = 80

    #######################
    # Start Simulation in Synchronous mode
    #######################
    vrep.simxSynchronous(clientID,True)
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)

    #######################
    # Set position of vehicle
    #######################
    veh_y = genStraight( lane_width, lane_len, obs_w, i*case_width, i, getInit = True )
    err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[ i*case_width, veh_y, 0.2],vrep.simx_opmode_oneshot)
    err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle,-1,[0,0,math.radians(90)],vrep.simx_opmode_oneshot)

    
    #######################
    # Create Dummy for organizing each lane
    #######################
    createDummy([0,0,10], 'Scene' + str(i), parentHandle=-1)
    err_code, scene_handle = vrep.simxGetObjectHandle( clientID, "Scene" + str(i), vrep.simx_opmode_blocking) 

    #######################
    # Gen Case
    #######################
    genTee( lane_width, turn_len, lane_len, obs_w, i*case_width, i, direction = i % 3, openWall = i % 2)

    # Set parent of vehicle
    vrep.simxSetObjectParent(clientID,vehicle_handle, scene_handle, True, vrep.simx_opmode_blocking)

    # Create dummy
    #vrep.simxCreateDummy( clientID, 1, None, vrep.simx_opmode_blocking)
    #err_code, dummy_handle = vrep.simxGetObjectHandle( clientID, "Dummy" + str(i), vrep.simx_opmode_blocking) 
    #ret_code = vrep.simxSetObjectPosition( clientID, dummy_handle, -1, [0 + i*20,60,0.3], vrep.simx_opmode_blocking)
    vrep.simxPauseSimulation(clientID,vrep.simx_opmode_blocking)
    time.sleep(1)




# Now close the connection to V-REP:
vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
vrep.simxFinish(clientID)

print ('Program ended')
