import numpy as np
import math   
import pickle
import sys
from utils.scene_constants_pb import scene_constants
from icecream import ic
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# from utils.scene_constants_pb import scene_constants
# import scene_constants_pb





# Generate Trajectory given vehicle information

# Inputs
#   options           : given
#   scene_const       : given
#   next_veh_pos      : VEH_COUNT x 2
#               UNITS :     meter
#           STRUCTURE :     [x1, y1; x2, y2; x3, y3...]
#   next_veh_heading  : VEH_COUNT x 1
#               UNITS :     radians
#           STRUCTURE :     0 for facing north. CCW : +, CW -    
#   next_state_sensor : VEH_COUNT x SENSOR_COUNT*2 x FRAME_COUNT
#               UNITS :     Normlized (0-1) & Binary (0 or 1)
#           STRUCTURE :     (:, 0 ~ sensor_count-1, :) - distance
#           STRUCTURE :     (:, sensor_count ~ 2*sensor_count-1, :) - sensor state
#   next_state_goal   : VEH_COUNT x 2 (heading,distance) x FRAME_COUNT
#               UNITS :     heading : radians / distance : normalized (0-1)
#   network_model     : keras network
#   max_horizon       : integer
#   debug : T/F, If true, then prints additional information

# Outputs
#   trajectory        : 2(x,y)^T x max_horizon 

def genTrajectory(options, scene_const, next_veh_pos, next_veh_heading, next_state_sensor, next_state_goal, network_model, max_horizon, debug = False):
    # Set curr variable
    oldPosition = next_veh_pos
    oldHeading  = next_veh_heading
    oldSensor0   = next_state_sensor
    # oldPosition = oldPosition[:, :, options.FRAME_COUNT - 1]
    # oldHeading  = oldHeading[:, 2, options.FRAME_COUNT - 1]
    oldSensor   = oldSensor0[:, :, options.FRAME_COUNT - 1]
    oldGoal     = next_state_goal

    # Define variable for estimations of new sensor info
    newStateStack    = np.zeros([options.VEH_COUNT, scene_const.sensor_count*2, max_horizon])
    newGoalStack     = np.zeros([options.VEH_COUNT, 2, max_horizon])
    newPositionStack = np.zeros([options.VEH_COUNT, 2, max_horizon])
    newHeadingStack  = np.zeros([options.VEH_COUNT, 1, max_horizon])
    newSensorXStack = np.zeros([options.VEH_COUNT, scene_const.sensor_count, max_horizon])
    newSensorYStack = np.zeros([options.VEH_COUNT, scene_const.sensor_count, max_horizon])

    if debug == True:
        ic('Main Loop for Computing Prediction...')

    prevPosition    = oldPosition
    prevHeading     = oldHeading
    for t in range(0,max_horizon):
        if debug == True:
            ic('Horizon', t)
            # ic(oldSensor, oldSensor.shape)

        # Input to the network
        action_feed = {}
        action_feed.clear()
        action_feed.update({'observation_sensor_k': oldSensor0[:,0:scene_const.sensor_count,:]})
        action_feed.update({'observation_state': oldSensor0[:,scene_const.sensor_count:,:]})
        action_feed.update({'observation_goal_k': oldGoal})

        # Dummy input for action (this is not used)
        action_feed.update({'action_k': np.zeros((options.VEH_COUNT,options.ACTION_DIM))})

        # targetSteer_k, action_stack_k = getOptimalAction( action_feed ) # FIXME

        # Get Optimal Action
        act_values = network_model.predict(action_feed, batch_size=options.VEH_COUNT)

        # Get maximum for each vehicle
        action_stack_k = np.argmax(act_values, axis=1)

        # Apply the Steering Action & Keep Velocity. For some reason, +ve means left, -ve means right
        targetSteer_k = math.radians( scene_const.min_steer + action_stack_k * abs(scene_const.max_steer - scene_const.min_steer)/(options.ACTION_DIM-1) )

        if debug == True:
            ic(act_values)
            ic(action_stack_k)
            ic(targetSteer_k)

        # Update curr_state using dynamics model
        newPosition, newHeading = getVehicleEstimation(options, prevPosition, prevHeading, targetSteer_k)

        # Estimate of LIDAR distance / LIDAR detection
        newSensor, newSensorState, newSensorXY = predictLidar(options, scene_const, oldPosition, oldHeading, oldSensor, newPosition, newHeading)

        # Combine sensor values into single variable
        newSensorCombined = np.concatenate((newSensor,newSensorState), axis=1)

        # Add new data to oldSensor0
        temp = np.zeros((options.VEH_COUNT,scene_const.sensor_count*2,options.FRAME_COUNT))
        temp[:,:,0:options.FRAME_COUNT-1]     = oldSensor0[:,:,1:]
        temp[:,:,options.FRAME_COUNT-1]       = newSensorCombined
        oldSensor0 = temp
        # oldSensor0 = np.concatenate((oldSensor0, newSensorCombined)) 


        # Estimate goal angle & distance
        newGoal = getGoalEstimation(options, scene_const, newPosition, newHeading, oldGoal[:,:,options.FRAME_COUNT-1])
        oldGoal[:,:,0:options.FRAME_COUNT-1] = oldGoal[:,:,1:options.FRAME_COUNT]
        oldGoal[:,:,options.FRAME_COUNT-1]   = newGoal

        # Save data for plotting
        newGoalStack[:,:,t]         = newGoal 
        newStateStack[:, :, t]      = newSensorCombined
        newPositionStack[:, :, t]   = newPosition
        newHeadingStack[:, :, t]    = newHeading
        newSensorXStack[:, :, t] = newSensorXY[:,:,0]
        newSensorYStack[:, :, t] = newSensorXY[:,:,1,]

        prevPosition = newPosition
        prevHeading  = newHeading

        if debug == True:
            print("\n")

    # Return the vehicle trajectory
    return newPositionStack, newStateStack, newHeadingStack, newSensorXStack, newSensorYStack

# Get estimated position and heading of vehicle
# Input
#   options         : given
#   oldPosition     : VEH_COUNT x 2
#   oldHeading      : VEH_COUNT x 1 (angle)
#   targetSteer_k   : VEH_COUNT x 1 (each value mean steering angle)
# Output
#   newPosition     : VEH_COUNT x 2
#   newHeading      : VEH_COUNT x 1
def getVehicleEstimation(options, oldPosition, oldHeading, targetSteer_k):
    # FIXME: velocity scaling
    vLength = 0.8
    vel     = options.INIT_SPD*0.05
    delT    = (1/60)*options.FIX_INPUT_STEP

    newPosition = np.zeros([options.VEH_COUNT,2])
    newHeading  = np.zeros([options.VEH_COUNT])

    newPosition[:,0] = oldPosition[:,0] + vel * np.sin(oldHeading)*delT
    newPosition[:,1] = oldPosition[:,1] + vel * np.cos(oldHeading)*delT
    newHeading       = oldHeading + vel/vLength*np.tan(targetSteer_k)*delT

    # for v in range(0,options.VEH_COUNT):
        # newPosition[v,0]=oldPosition[v,0]+vel*np.cos(oldHeading[v])*delT
        # newPosition[v,1]=oldPosition[v,1]+vel*np.sin(oldHeading[v])*delT
        # newHeading[v]=oldHeading[v]+vel/vLength*math.tan(targetSteer_k[v])*delT

    return newPosition, newHeading

# Get estimated goal length and heading
# Input
#   options
#   scene_const
#   veh_pos : VEH_COUNT x 2
#   veh_heading : VEH_COUNT
#   g_pos : VEH_COUNT x 2 (x,y) position of goal point 
def getGoalEstimation(options, scene_const, veh_pos, veh_heading, g_pos):
    gInfo           = np.zeros([options.VEH_COUNT,2])
    # goal_handle     = handle_dict['dummy']     # FIXME

    for k in range(0,options.VEH_COUNT):
        # To compute gInfo, get goal position
        # g_pos, _ = p.getBasePositionAndOrientation( goal_handle[k] )

        # Calculate the distance
        # ic(g_pos[0:2],veh_pos[k])
        delta_distance  = np.array(g_pos[k]) - np.array(veh_pos[k])  # delta x, delta y
        gInfo[k][1]     = np.linalg.norm(delta_distance) / scene_const.goal_distance

        # calculate angle. 90deg + angle (assuming heading north) - veh_heading
        gInfo[k][0] = math.atan(abs(delta_distance[0]) / abs(delta_distance[1]))  # delta x / delta y, and scale by pi 1(left) to -1 (right)
        if delta_distance[0] < 0:
            # Goal is left of the vehicle, then -1
            gInfo[k][0] = gInfo[k][0] * -1

        # Scale with heading
        gInfo[k][0] = -1 * (math.pi * 0.5 - gInfo[k][0] - veh_heading[k]) / (math.pi / 2)

    return gInfo

# Predict next LIDAR state
# Input
#   options
#   scene_const
#   oldPosition : VEH_COUNT x 2
#   oldHeading  : VEH_COUNT
#   oldSensor   : VEH_COUNT x SENSOR_COUNT*2 x FRAME_COUNT
#   newPosition : VEH_COUNT x 2
#   newHeading  : VEH_COUNT
# Output
#   newSensor   : VEH_COUNT x SENSOR_COUNT
#   newSensorState : VEH_COUNT x SENSOR_COUNT
#   newSensorXY : VEH_COUNT X SENSOR_COUNT X 2
def predictLidar(options, scene_const, oldPosition, oldHeading, oldSensor, newPosition, newHeading):
    oldSensorState  = oldSensor[:,scene_const.sensor_count:]
    # oldSensor=oldSensor[:,0:scene_const.sensor_count-1]
    oldSensor       = oldSensor[:,0:scene_const.sensor_count]

    seqAngle        = np.array([(scene_const.sensor_count-1-i)*np.pi/(scene_const.sensor_count-1) for i in range(scene_const.sensor_count)])
    newSensor       = np.zeros((options.VEH_COUNT,scene_const.sensor_count))
    newSensorState  = np.zeros((options.VEH_COUNT,scene_const.sensor_count)) # 0:closed & 1:open
    newSensorXY      = np.zeros((options.VEH_COUNT,scene_const.sensor_count,2))

    for v in range(0,options.VEH_COUNT):
        # oldC,oldS   = np.cos(oldHeading[v]-np.pi/2),np.sin(oldHeading[v]-np.pi/2)
        oldC,oldS   = np.cos(-oldHeading[v]),np.sin(-oldHeading[v])
        oldRot      = np.array([[oldC,-oldS],[oldS,oldC]])
        # newC,newS   = np.cos(newHeading[v]-np.pi/2),np.sin(newHeading[v]-np.pi/2)
        newC,newS   = np.cos(-newHeading[v]+oldHeading[v]),np.sin(-newHeading[v]+oldHeading[v])
        newRot      = np.array([[newC,-newS],[newS,newC]])

        # Transformation oldSensor w.r.t vehicle's next position and heading.
        temp1   = scene_const.sensor_distance*np.transpose([oldSensor[v],oldSensor[v]])
        temp2   = np.transpose([np.cos(seqAngle), np.sin(seqAngle)])
        oldLidarXY = np.multiply(temp1,temp2)
        #oldLidarXY = np.tile(oldPosition[v, :], [scene_const.sensor_count, 1]) \
        #            + np.matmul(oldLidarXY, oldRot)
        oldTransXY = oldLidarXY-np.tile(newPosition[v,:], [scene_const.sensor_count,1])
        oldTransXY = np.matmul(oldTransXY,newRot)

        # Compute newSensor w.r.t vehicle's next position and heading
        newTransXY = scene_const.sensor_distance*np.transpose([np.cos(seqAngle), np.sin(seqAngle)])

        # Remove out-range points, i.e. y<0
        reducedLidarXY=np.array([])
        for i in range(0,scene_const.sensor_count):
            if oldTransXY[i,1]>=0:
                reducedLidarXY = np.append(reducedLidarXY, oldTransXY[i,:])
        length=reducedLidarXY.shape[0]
        reducedLidarXY=np.reshape(reducedLidarXY, (-1,2))

        # Find intersection between line1(newPosition,newTransXY) & line2(two among reducedLidarXY)
        newLidarXY=np.zeros([scene_const.sensor_count,2])
        flag=np.ones([scene_const.sensor_count])
        length=reducedLidarXY.shape[0]
        for i in range(0,scene_const.sensor_count):
            for j in range(0,length-1):
                l=j+1
                # If intersection is found, then pass
                if flag[i]==0:
                    pass

                A=np.array([[newTransXY[i][1],-newTransXY[i][0]],[reducedLidarXY[l][1]-reducedLidarXY[j][1],-reducedLidarXY[l][0]+reducedLidarXY[j][0]]])
                det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
                if det==0:
                    pass
                b=np.array([0,np.dot(A[1,:],reducedLidarXY[j,:])])
                try:
                    x=np.linalg.solve(A,b)
                except:
                    pass

                # check the point between two pair points
                # if np.dot(A[1,:],newTransXY[i]-x)*np.dot(A[1,:],-x)<=0 \
                #         and np.dot(A[0,:],reducedLidarXY[j]-x)*np.dot(A[0,:],reducedLidarXY[l]-x)<=0:
                if np.dot(A[0, :], reducedLidarXY[j] - x) * np.dot(A[0, :], reducedLidarXY[l] - x) <= 0:
                    newLidarXY[i,:]=x
                    newSensorState[v,i]=oldSensorState[v,j]*oldSensorState[v,l] # closed if there exist at least closed point
                    flag[i]=0

        # Compute newSensorOut
        newLidarXY=newLidarXY+1.5*scene_const.collision_distance*np.transpose([np.cos(seqAngle),np.sin(seqAngle)])*np.transpose([flag,flag])
        newSensor[v,:]=np.linalg.norm(newLidarXY,ord=2,axis=1)/scene_const.sensor_distance

        newLidarXY = np.matmul(newLidarXY, np.transpose(newRot))
        #newLidarXY = newLidarXY + np.tile(newPosition[v, :]-oldPosition[v, :], [scene_const.sensor_count, 1])
        newLidarXY = np.matmul(newLidarXY, np.transpose(oldRot))
        #newLidarXY = newLidarXY - np.tile(newPosition[v, :], [scene_const.sensor_count, 1])
        newSensorXY[v,:,:]=newLidarXY

    return newSensor, newSensorState, newSensorXY




# Initilize the generate trajecotry
# Inputs
#   file: relative path to file storing options & scene_const
# Outputs
#   options
#   scene_const
#   model

def genTrajectoryInit( weightFilePath, optionFilePath = 'genTraj_options_file'):
    # Load options/scene_const file
    infile = open( optionFilePath ,'rb')
    new_dict = pickle.load(infile)
    infile.close()

    # Variables
    sample_options     = new_dict['options']
    sample_scene_const = new_dict['scene_const']

    print('=========================================')
    print(str(sample_options))
    print('=========================================')
    print(str(sample_scene_const))
    print('=========================================')

    # Load the full model
    model = tf.keras.models.load_model( weightFilePath, custom_objects={"tf": tf} )
    model.summary()

    # Load the model which outputs Q-values for all action
    model_q_all = tf.keras.Model( inputs = model.input, outputs = model.get_layer('out_large').output ) 
    model_q_all.summary()


    # for layer in model.layers:
        # print( layer.get_weights() )


    return sample_options, sample_scene_const, model_q_all


#######################
# EXAMPLE 
#######################
# Following is the example usage of the functions defined in this file.
#######################
# np.set_printoptions(linewidth = 100)

# weightFilePath = './model_weights/checkpoints-vehicle-CNN-state-191108/2019-11-07_19_55_36.380100_e5000_gs369576.h5'
# optionFilePath = './model_weights/checkpoints-vehicle-CNN-state-191108/genTraj_options_file'

# # Load options & Network
# sample_options, sample_scene_const, network_model = genTrajectoryInit( weightFilePath, optionFilePath )

# sample_veh_pos      = np.zeros((sample_options.VEH_COUNT,2))
# sample_veh_heading  = np.zeros(sample_options.VEH_COUNT)
# sample_state_sensor = np.ones((sample_options.VEH_COUNT, sample_scene_const.sensor_count*2, sample_options.FRAME_COUNT))*0.5
# sample_state_goal   = np.ones((sample_options.VEH_COUNT, 2, sample_options.FRAME_COUNT))*1
# sample_state_goal[0][1][:] = 0  # set goal angle to 0, travel straight
# max_horizon         = 5

# DEBUG = True

# if DEBUG == True:
#     ic(sample_veh_pos)
#     ic(sample_veh_heading)
#     ic(sample_state_sensor)
#     ic(sample_state_goal)
#     print('\n\n')


# traj_est, _ = genTrajectory(sample_options, sample_scene_const, sample_veh_pos, sample_veh_heading, sample_state_sensor, sample_state_goal, network_model, max_horizon, debug = DEBUG)

# ic(traj_est)

# fig = plt.figure()
# plt.plot(traj_est[0][0][:], traj_est[0][1][:])

# plt.show()