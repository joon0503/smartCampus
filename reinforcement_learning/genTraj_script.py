import numpy as np
import math   
import pickle
import sys
from utils.scene_constants_pb import scene_constants
from icecream import ic
import tensorflow as tf


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

# Outputs
#   trajectory        : 2(x,y)^T x max_horizon 

def genTrajectory(options, scene_const, next_veh_pos, next_veh_heading, next_state_sensor, next_state_goal, network_model, max_horizon):
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

    prevPosition    = oldPosition
    prevHeading     = oldHeading
    for t in range(0,max_horizon):
        ic(oldSensor, oldSensor.shape)

        action_feed = {}
        action_feed.clear()
        action_feed.update({'observation_sensor_k': oldSensor0[:,0:scene_const.sensor_count,:]})
        action_feed.update({'observation_state': oldSensor0[:,scene_const.sensor_count:,:]})
        action_feed.update({'observation_goal_k': oldGoal})
        # targetSteer_k, action_stack_k = getOptimalAction( action_feed ) # FIXME

        # Get Optimal Action
        act_values = network_model.predict(action_feed, batch_size=options.VEH_COUNT)

            # Get maximum for each vehicle
        action_stack_k = np.argmax(act_values, axis=1)

            # Apply the Steering Action & Keep Velocity. For some reason, +ve means left, -ve means right
        targetSteer_k = scene_const.max_steer - action_stack_k * abs(scene_const.max_steer - scene_const.min_steer)/(options.ACTION_DIM-1)



        # Update curr_state using dynamics model
        newPosition, newHeading = getVehicleEstimation(options, prevPosition, prevHeading, targetSteer_k)

        # Estimate of LIDAR distance / LIDAR detection
        newSensor, newSensorState = predictLidar(options, scene_const, oldPosition, oldHeading, oldSensor, newPosition, newHeading)

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
        
        prevPosition = newPosition
        prevHeading  = newHeading

    # Return the vehicle trajectory
    return newPositionStack

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
    vLength = 2.1
    vel     = options.INIT_SPD
    delT    = options.CTR_FREQ*options.FIX_INPUT_STEP

    newPosition = np.zeros([options.VEH_COUNT,2])
    newHeading  = np.zeros([options.VEH_COUNT])

    newPosition[:,0] = oldPosition[:,0] + vel * np.cos(oldHeading)*delT
    newPosition[:,1] = oldPosition[:,1] + vel * np.sin(oldHeading)*delT
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
def predictLidar(options, scene_const, oldPosition, oldHeading, oldSensor, newPosition, newHeading):
    oldSensorState  = oldSensor[:,scene_const.sensor_count:]
    # oldSensor=oldSensor[:,0:scene_const.sensor_count-1]
    oldSensor       = oldSensor[:,0:scene_const.sensor_count]

    seqAngle        = np.array([(scene_const.sensor_count-1-i)*np.pi/(scene_const.sensor_count-1) for i in range(scene_const.sensor_count)])
    newSensor       = np.zeros((options.VEH_COUNT,scene_const.sensor_count))
    newSensorState  = np.ones((options.VEH_COUNT,scene_const.sensor_count)) # 0:closed & 1:open

    for v in range(0,options.VEH_COUNT):
        oldC,oldS   = np.cos(oldHeading[v]-np.pi/2),np.sin(oldHeading[v]-np.pi/2)
        oldRot      = np.array([[oldC,oldS],[-oldS,oldC]])
        newC,newS   = np.cos(newHeading[v]-np.pi/2),np.sin(newHeading[v]-np.pi/2)
        newRot      = np.array([[newC,newS],[-newS,newC]])

        # Transformation oldSensor w.r.t vehicle's next position and heading.
        temp1   = scene_const.sensor_distance*np.transpose([oldSensor[v],oldSensor[v]])
        temp2   = np.transpose([np.cos(seqAngle), np.sin(seqAngle)])
        oldLidarXY = np.multiply(temp1,temp2)
        oldLidarXY = np.tile(oldPosition[v, :], [scene_const.sensor_count, 1]) \
                    + np.matmul(oldLidarXY, np.transpose(oldRot))
        oldTransXY = oldLidarXY-np.tile(newPosition[v,:], [scene_const.sensor_count,1])
        oldTransXY = np.matmul(oldTransXY,np.transpose(newRot))

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

    return newSensor, newSensorState






# Initilize the generate trajecotry
# Inputs
#   file: relative path to file storing options & scene_const
# Outputs
#   options
#   scene_const
#   model

def genTrajectoryInit( weightFilePath, optionFilePath = 'genTraj_options_file' ):
    # Load options/scene_const file
    infile = open( optionFilePath ,'rb')
    new_dict = pickle.load(infile)
    infile.close()

    # Variables
    sample_options     = new_dict['options']
    sample_scene_const = new_dict['scene_const']

    # Load model
    model = tf.keras.models.load_model( weightFilePath, custom_objects={"tf": tf} )

    model.summary()

    # for layer in model.layers:
        # print( layer.get_weights() )


    return sample_options, sample_scene_const, model


#######################
# EXAMPLE 
#######################

# weightFilePath = './checkpoints-vehicle/2019-09-30_11_17_17.948153_e25_gs1674.h5'

# # Load options & Network
# sample_options, sample_scene_const, network_model = genTrajectoryInit( weightFilePath )

# sample_veh_pos      = np.zeros((sample_options.VEH_COUNT,2))
# sample_veh_heading  = np.zeros(sample_options.VEH_COUNT)
# sample_state_sensor = np.ones((sample_options.VEH_COUNT, sample_scene_const.sensor_count*2, sample_options.FRAME_COUNT))
# sample_state_goal   = np.ones((sample_options.VEH_COUNT, 2, sample_options.FRAME_COUNT))*100
# max_horizon         = 5

# traj_est = genTrajectory(sample_options, sample_scene_const, sample_veh_pos, sample_veh_heading, sample_state_sensor, sample_state_goal, network_model, max_horizon)