########
# This file generate the scene file for Reinforcedment learning
# You must open vrep, set simulation option such that it keeps newly added objects.
# Also, it requires dyros_vehicle to be present
# Then run this script.

# This particular script generates lane-change scenario with obstacle.
#######



###############
# Helper
###############

# Create Object
# objSize: size of object, triple
# objPos : position of object, triple
# objName: name of object, string
def createObject(objSize, objPos, objName):
    # First remove object if one exists with same name
    retCode, retHandle = vrep.simxGetObjectHandle(clientID, objName, vrep.simx_opmode_blocking)

    # retCode == 0 means function executed, i.e., there is already a another object
    if retCode == 0:
        print(objName + " already exists. It will be removed first.")
        vrep.simxRemoveObject(clientID, retHandle, vrep.simx_opmode_blocking)

    emptyBuff = bytearray()
    inputFloats = objSize + objPos #concatenation

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'createCuboid_function',[],inputFloats,[objName],emptyBuff,vrep.simx_opmode_blocking)
    if res==vrep.simx_return_ok:
        print ( objName + ' created!')
    else:
        print ('Creating object failed.')

    return

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
    
# Position dyros_vehicle
err_code,vehicle_handle = vrep.simxGetObjectHandle(clientID,"dyros_vehicle", vrep.simx_opmode_blocking) 
err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[0,0,0.2],vrep.simx_opmode_oneshot)
err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle,-1,[0,0,math.radians(90)],vrep.simx_opmode_oneshot)

# Create Sensors
SENSOR_COUNT = 5
RAD_DT = math.pi/(SENSOR_COUNT-1)

sensor_handle_array = [0]
err_code,sensor_handle_array[0] = vrep.simxGetObjectHandle(clientID,"Proximity_sensor", vrep.simx_opmode_blocking)
if err_code != 0:
   errorExit() 

for i in range(0,SENSOR_COUNT):
    print(str(i))
    err_code1, sensor_handle = vrep.simxCopyPasteObjects(clientID,[sensor_handle_array[0]], vrep.simx_opmode_blocking)
    vrep.simxSetObjectParent(clientID,sensor_handle[0],vehicle_handle,False,vrep.simx_opmode_blocking)
    ret_code = vrep.simxSetObjectPosition(clientID,sensor_handle[0],vehicle_handle,[2,0,0.7],vrep.simx_opmode_blocking)
    vrep.simxSetObjectOrientation(clientID,sensor_handle[0],vehicle_handle,[-math.pi/2,RAD_DT*i,0],vrep.simx_opmode_oneshot)
    sensor_handle_array = np.append(sensor_handle_array,sensor_handle)   

# Start Simulation in Synchronous mode
vrep.simxSynchronous(clientID,True)
vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)

# Create road and walls
createObject([5, 150, 0.1],[-1.25,0,-0.1],'floor')                  # create floor
createObject([0.1, 150, 2.5],[-3.75,0,1.25],'wallLeft')             # create wall left
createObject([0.1, 150, 2.5],[1.25,0,1.25],'wallRight')             # create wall right
createObject([2.2, 2.2 , 2],[0,30, 1.1],'obstacle')                 # create obstacle

# Create dummy
vrep.simxCreateDummy( clientID, 1, None, vrep.simx_opmode_blocking)
err_code, dummy_handle = vrep.simxGetObjectHandle( clientID, "Dummy", vrep.simx_opmode_blocking) 
ret_code = vrep.simxSetObjectPosition( clientID, dummy_handle, -1, [0,60,0.3], vrep.simx_opmode_blocking)

# Now close the connection to V-REP:
vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
vrep.simxFinish(clientID)

print ('Program ended')
