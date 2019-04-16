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
  

COPY_NUM = 20
SENSOR_COUNT = 9

err_code,dyros_handle = vrep.simxGetObjectHandle(clientID,"dyros_vehicle", vrep.simx_opmode_blocking) 

for i in range(0,COPY_NUM):
    # create vehicle
    err_code1, vehicle_handle = vrep.simxCopyPasteObjects(clientID,[dyros_handle], vrep.simx_opmode_blocking)
    vehicle_handle = vehicle_handle[0]

    # Position dyros_vehicle
#    err_code,vehicle_handle = vrep.simxGetObjectHandle(clientID,"dyros_vehicle0", vrep.simx_opmode_blocking) 
    print(i*20)

    # Create Sensors
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
        ret_code = vrep.simxSetObjectPosition(clientID,sensor_handle[0],vehicle_handle,[1.5,0,0.7],vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(clientID,sensor_handle[0],vehicle_handle,[-math.pi/2,RAD_DT*s,0],vrep.simx_opmode_oneshot)
        sensor_handle_array = np.append(sensor_handle_array,sensor_handle)   



    # Start Simulation in Synchronous mode
    vrep.simxSynchronous(clientID,True)
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    
    # Set position of vehicle
    err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[0 + i*20,0,0.2],vrep.simx_opmode_oneshot)
    err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle,-1,[0,0,math.radians(90)],vrep.simx_opmode_oneshot)

    # Create Dummy for organizing each lane
    createDummy([0,0,10], 'Scene' + str(i), parentHandle=-1)
    err_code, scene_handle = vrep.simxGetObjectHandle( clientID, "Scene" + str(i), vrep.simx_opmode_blocking) 

    # Create Goal point
    createDummy([0+i*20,60,0.3], 'GoalPoint' + str(i), parentHandle=scene_handle)

    # Create road and walls
    lane_width = 12

    createObject([lane_width, 150, 0.1],[-1.25 + i*20,0,-0.1],'floor' + str(i), scene_handle)                  # create floor
    createObject([0.1, 150, 2.5],[-5 + i*20,0,1.25],'wallLeft' + str(i), scene_handle)             # create wall left
    createObject([0.1, 150, 2.5],[3 + i*20,0,1.25],'wallRight' + str(i), scene_handle)             # create wall right
    if i % 2 == 0:
        createObject([3, 5 , 2],[1.5 + i*20,30, 1.1],'obstacle' + str(i), scene_handle)                 # create obstacle on the right
    else:
        createObject([3, 5 , 2],[1.5 + i*20 - 5.0,30, 1.1],'obstacle' + str(i), scene_handle)                 # create obstacle on the left
        
    createObject([lane_width, 0.2 , 2.5],[-1.25 + i*20,70, 1.25],'wallEnd' + str(i), scene_handle)                 # create end wall
    createObject([lane_width, 0.2 , 2.5],[-1.25 + i*20,-10, 1.25],'wallBack' + str(i), scene_handle)                 # create end wall

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
