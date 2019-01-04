# DQN model to solve vehicle control problem

import tensorflow as tf
import random
import numpy as np
import math
import sys
import vrep
import matplotlib
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import os
import re
from collections import deque
from experience_replay import SumTree
from experience_replay import Memory
from argparse import ArgumentParser

MAX_DISTANCE = 20
COLLISION_DISTANCE = 1.3
GOAL_DISTANCE = 60
SENSOR_COUNT = 9

def get_options():
    parser = ArgumentParser(
        description='File for learning'
        )
    parser.add_argument('--MAX_EPISODE', type=int, default=10001,
                        help='max number of episodes iteration\n')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=500,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=11,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.99,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=1.0,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-1,
                        help='finial probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.995,
                        help='epsilon decay rate')
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=2000,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=20000,
                        help='size of experience replay memory')
    parser.add_argument('--SAVER_RATE', type=int, default=20000,
                        help='Save network after this number of episodes')
    parser.add_argument('--FIX_INPUT_STEP', type=int, default=4,
                        help='Fix chosen input for this number of steps')
    parser.add_argument('--TARGET_UPDATE_STEP', type=int, default=3000,
                        help='Number of steps required for target update')
    parser.add_argument('--BATCH_SIZE', type=int, default=32,
                        help='mini batch size'),
    parser.add_argument('--H1_SIZE', type=int, default=80,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=80,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=80,
                        help='size of hidden layer 3')
    parser.add_argument('--RESET_STEP', type=int, default=10000,
                        help='number of episode after resetting the simulation')
    parser.add_argument('--RUNNING_AVG_STEP', type=int, default=100,
                        help='number of episode to calculate the average score')
    parser.add_argument('--manual','-m', action='store_true',
                        help='Step simulation manually')
    parser.add_argument('--NO_SAVE','-us', action='store_true',
                        help='Use saved tensorflow network')
    parser.add_argument('--TESTING','-t', action='store_true',
                        help='No training. Just testing. Use it with eps=1.0')
    parser.add_argument('--disable_DN', action='store_true',
                        help='Disable the usage of double network.')
    parser.add_argument('--disable_PER', action='store_true',
                        help='Disable the usage of PER.')
    parser.add_argument('--disable_duel', action='store_true',
                        help='Disable the usage of double network.')
    parser.add_argument('--FRAME_COUNT', type=int, default=1,
                        help='Number of frames to be used')
    parser.add_argument('--ACT_FUNC', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--GOAL_REW', type=int, default=1000,
                        help='Activation function')
    parser.add_argument('--FAIL_REW', type=int, default=-1000,
                        help='Activation function')
    parser.add_argument('--VEH_COUNT', type=int, default=10,
                        help='Number of vehicles to use for simulation')
    parser.add_argument('--INIT_SPD', type=int, default=10,
                        help='Initial speed of vehicle in  km/hr')
    parser.add_argument('--DIST_MUL', type=int, default=10,
                        help='Multiplier for rewards based on the distance to the goal')
    options = parser.parse_args()
    return options


# Class for Neural Network
class QAgent:
    def __init__(self, options, name):
        self.scope = name
    
        # Parse input for activation function
        if options.ACT_FUNC == 'elu':
            act_function = tf.nn.elu
        elif options.ACT_FUNC == 'relu':
            act_function = tf.nn.relu
        else:
            raise NameError('Supplied activation function is not supported!')
            return 

        with tf.variable_scope(self.scope):      # Set variable scope

            ######################################:
            ## CONSTRUCTING NEURAL NETWORK
            ######################################:

            # Inputs
#            self.sensor_input   = tf.placeholder(tf.float32, [None, SENSOR_COUNT*options.FRAME_COUNT], name='observation')
#            self.goal_input     = tf.placeholder(tf.float32, [None, 2*options.FRAME_COUNT], name='observation')
            self.observation    = tf.placeholder(tf.float32, [None, (SENSOR_COUNT+2)*options.FRAME_COUNT], name='observation')
            self.ISWeights      = tf.placeholder(tf.float32, [None,1], name='IS_weights')
            self.act            = tf.placeholder(tf.float32, [None, options.ACTION_DIM],name='action')
            self.target_Q       = tf.placeholder(tf.float32, [None, ], name='target_q' )  


            if False:
                # Slicing
                self.sensor_data    = tf.slice(self.observation, [0, 0], [-1, SENSOR_COUNT*options.FRAME_COUNT])
                self.goal_data      = tf.slice(self.observation, [0, SENSOR_COUNT*options.FRAME_COUNT], [-1, 2])

                # "CNN-like" structure for sensor data. 2 Layers
                self.h_s1 = tf.layers.dense( inputs=self.sensor_data,
                                             units=options.H1_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="h_s1"
                                           )
     
                self.h_s2 = tf.layers.dense( inputs=self.h_s1,
                                             units=options.H1_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="h_s2"
                                           )

                # Combine sensor data and goal data
                self.combined_layer = tf.concat([self.goal_data, self.h_s2], -1)

                # FC Layer
                self.h_fc_1 = tf.layers.dense( inputs=self.combined_layer,
                                             units=options.H1_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="h_fc1"
                                           )

                if options.disable_duel == True:
                    # Regular DQN
                    self.h_fc_2 = tf.layers.dense( inputs=self.h_fc_1,
                                                 units=options.H1_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="h_fc2"
                                               )
                    
                    self.output = tf.layers.dense( inputs=self.h_fc_2,
                                                 units=options.ACTION_DIM,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="q_value"
                                               )
                else:
                    # Dueling Network
                    self.h_layer_val = tf.layers.dense( inputs=self.h_fc_1,
                                                 units=options.H3_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="h_layer_val"
                                               )
                    self.h_layer_adv = tf.layers.dense( inputs=self.h_fc_1,
                                                 units=options.H3_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="h_layer_adv"
                                               )

                    # Value and advantage estimate
                    self.val_est    = tf.layers.dense( inputs=self.h_layer_val,
                                                 units=1,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="val_est"
                                               )

                    self.adv_est    = tf.layers.dense( inputs=self.h_layer_adv,
                                                 units=options.ACTION_DIM,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="adv_est"
                                               )
                 
                    self.output = self.val_est + tf.subtract(self.adv_est, tf.reduce_mean(self.adv_est, axis=1, keepdims=True) ) 
            else:
                print("======================")
                print("Regular Net")
                print("======================")
                self.sensor_data    = tf.slice(self.observation, [0, 0], [-1, SENSOR_COUNT*options.FRAME_COUNT])
                self.goal_data      = tf.slice(self.observation, [0, SENSOR_COUNT*options.FRAME_COUNT], [-1, 2])

                # Regular neural net
                self.h_s1 = tf.layers.dense( inputs=self.observation,
                                             units=options.H1_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             #kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1)
                                             name="h_s1"
                                           )
                self.h_s2 = tf.layers.dense( inputs=self.h_s1,
                                             units=options.H2_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="h_s2"
                                           )
                self.h_s3 = tf.layers.dense( inputs=self.h_s2,
                                             units=options.H3_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="h_s3"
                                           )
                self.output = tf.layers.dense( inputs=self.h_s3,
                                                  units=options.ACTION_DIM,
                                                  activation = None,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  name="h_action"
                                             )

            ######################################:
            ## END Constructing Neural Network
            ######################################:

            # Prediction of given specific action
            self.Q = tf.reduce_sum( tf.multiply(self.output, self.act), axis=1)

            # Absolute Loss
            self.loss_per_data = tf.abs(self.target_Q - self.Q, name='loss_per_data')    
            
            # Loss for Optimization
            self.loss = tf.reduce_mean(self.ISWeights * tf.square(self.target_Q - self.Q))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)


    # Sample action with random rate eps
    def sample_action(self, feed, eps, options, verbose = False):
        if random.random() <= eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = random.randrange(options.ACTION_DIM)
#            action = random.uniform(0,1)
        else:
            act_values = self.output.eval(feed_dict=feed)
            if options.TESTING == True or verbose == True:
                print(np.argmax(act_values))
                print(act_values)
            action_index = np.argmax(act_values)

        action = np.zeros(options.ACTION_DIM)
        action[action_index] = 1
        return action

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name


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

    # print("Desired Speed: " + str(desiredSpd) + " km/hr = " + str(desiredSpd_rps) + " radians per seconds = " + str(math.degrees(desiredSpd_rps)) + "degrees per seconds. = " + str(desiredSpd*(1000/3600)) + "m/s" )
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

def readSensor( clientID, sensorHandle, op_mode=vrep.simx_opmode_streaming ):
    dState      = []
    dDistance   = []
    for sHandle in sensorHandle:
        returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID, sHandle, op_mode)
        dState.append(detectionState)
        dDistance.append( np.linalg.norm(detectedPoint) )

    # Set false sensors to max distance
    for i in range(len(dDistance)):
        if dState[i] == 0:
            dDistance[i] = MAX_DISTANCE

    # change it into numpy int array
    dState =  np.array(dState)*1
    dDistance = np.array(dDistance)/MAX_DISTANCE 
    return dState.tolist(), dDistance.tolist()

# Detect whether vehicle is collided
# For now simply say vehicle is collided if distance if smaller than 1.1m. Initial sensor distance to the right starts at 1.2m.
# Input
#   dDistance: array of sensor measurement
#   dState   : array of sensor detections state
# Output
#   T/F      : whether collision occured
#   sensor # : Which sensor triggered collision. -1 if no collision
def detectCollision(dDistance):
    for i in range(SENSOR_COUNT):
        if dDistance[i]*MAX_DISTANCE < COLLISION_DISTANCE:
            return True, i
    
    return False, -1    

# Get goal point vector. Returns angle and distance
# Output
# goal_angle: angle to the goal point in radians / pi
#             Positive angle mean goal point is on the right of vehicle, negative mean goal point is on the left
# goal_distance: distance to the goal point / MAX_DISTANCE
def getGoalPoint( veh_index ):
    _, vehPos = vrep.simxGetObjectPosition( clientID, vehicle_handle[veh_index], -1, vrep.simx_opmode_blocking)            
    _, dummyPos = vrep.simxGetObjectPosition( clientID, dummy_handle[veh_index], -1, vrep.simx_opmode_blocking)            

    # Calculate the distance
    delta_distance = np.array(dummyPos) - np.array(vehPos)              # delta x, delta y
    goal_distance = np.linalg.norm(delta_distance)

    # calculate angle
    goal_angle = math.atan( delta_distance[0]/delta_distance[1] )       # delta x / delta y
    goal_angle = goal_angle / math.pi                               # result in -1 to 1

    return goal_angle, goal_distance / GOAL_DISTANCE

# veh_pos_info: (options.VEH_COUNT x 2)
def getGoalInfo( veh_pos_info, goal_pos_info ):
    # Calculate the distance
    delta_distance = goal_pos_info - veh_pos_info              # delta x, delta y
    #print(goal_pos_info)
    #print(veh_pos_info)
    #print(delta_distance)
    goal_distance = np.linalg.norm(delta_distance, axis=1)

    # calculate angle
    goal_angle = np.arctan( delta_distance[:,0]/delta_distance[:,1] )       # delta x / delta y
    goal_angle = goal_angle / math.pi                               # result in -1 to 1

    return goal_angle, goal_distance / GOAL_DISTANCE



# Detect whether vehicle reached goal point.
# Input
#   vehPos - [x,y,z] - center of vehicle. To get tip, add 2.075m
#   gInfo  - [angle,distance]
# Output
#   True/False
def detectReachedGoal(vehPos, gInfo, currHeading):
    # Distance less than 0.5m, angle less than 10 degrees
    if abs(gInfo[1]*GOAL_DISTANCE - 2.075) < 1.0 and abs(currHeading*90)<20: 
        return True
    else:
        return False


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
def getVehicleStateLUA():
    emptyBuff = bytearray()

    res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'remoteApiCommandServer',vrep.sim_scripttype_childscript,'getVehicleState_function',handle_list,[],[],emptyBuff,vrep.simx_opmode_blocking)

    # Unpack Data
    out_data = np.reshape(retFloats, (-1,18))       # Reshape such that each row is 18 elements. 3 for pos, 3 for ori, 9 for sensor, 3 for goal pos

    return out_data[:,0:2], ((out_data[:,5]-math.pi*0.5)/(math.pi/2)), out_data[:,6:6+SENSOR_COUNT]/MAX_DISTANCE, np.transpose( getGoalInfo( out_data[:,0:2], out_data[:,15:17] ) )


# Given one-hot-encoded action array of steering angle, apply it to the vehicle
# Input:
#   action: list of 1/0s. 1 means this action is applied
def applySteeringAction(action, veh_index):
    # Define maximum/minimum steering in degrees
    MAX_STEER = 15
    MIN_STEER = -15

    # Delta of angle between each action
    action_delta = (MAX_STEER - MIN_STEER) / (options.ACTION_DIM-1)

    # Calculate desired angle
    #action = [0,1,0,0,0]
    desired_angle = MAX_STEER - np.argmax(action) * action_delta

    # Set steering position
    setMotorPosition(clientID, steer_handle[veh_index], desired_angle)
    
    return


# Get Motor/Sensor Handles
# Input: clientID?
def getMotorHandles():
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

def getSensorHandles():
    # Get Sensor Handles
    sensor_handle = np.zeros([options.VEH_COUNT,SENSOR_COUNT], dtype=int)

    k = 0
    for v in range(0,options.VEH_COUNT):
        for i in range(0,SENSOR_COUNT):
            _,temp_handle  = vrep.simxGetObjectHandle(clientID, "Proximity_sensor" + str(k), vrep.simx_opmode_blocking)
            sensor_handle[v][i] = temp_handle
            k = k+1

    return sensor_handle

##################################
# TENSORFLOW HELPER
##################################

def printTFvars():
    print("==Training==")
    tvars = tf.trainable_variables(scope='Training')

    for var in tvars:
        print(var)

    print("==Target==")
    tvars = tf.trainable_variables(scope='Target')

    for var in tvars:
        print(var)
    print("=========")

    return

# Calculate the running average
# x: input 1D numpy array
# N: Window
def runningAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

##################################
# GENERAL
##################################

# From sensor and goal queue, get observation.
# observation is a row vector with all frames of information concatentated to each other
# Recall that queue stores 2*FRAME_COUNT of information. 
# first = True: get oldest info
# first = False: get latest info
def getObs( sensor_queue, goal_queue, old=True):

    if old == True:
        sensor_stack    = np.concatenate(sensor_queue)[0:SENSOR_COUNT*options.FRAME_COUNT]
        goal_stack      = np.concatenate(goal_queue)[0:2*options.FRAME_COUNT]
        observation     = np.concatenate((sensor_stack, goal_stack))    
    else:
        sensor_stack    = np.concatenate(sensor_queue)[SENSOR_COUNT*options.FRAME_COUNT:]
        goal_stack      = np.concatenate(goal_queue)[2*options.FRAME_COUNT:]
        observation     = np.concatenate((sensor_stack, goal_stack))    

    return observation

# Calculate Approximate Rewards for variaous cases
def printRewards():
    # Some parameters
    veh_speed = options.INIT_SPD/3.6  # 10km/hr in m/s

    # Time Steps
    dt = 0.025
    dt_code = dt * options.FIX_INPUT_STEP

    # Expected Total Time Steps
    total_step = (60/veh_speed)*(1/dt_code)

    # Reward at the end
    rew_end = 0
    for i in range(0, int(total_step)):
        goal_distance = 60 - i*dt_code*veh_speed 
        if i != total_step-1:
            rew_end = rew_end -options.DIST_MUL*(goal_distance/GOAL_DISTANCE)**2*(options.GAMMA**i) 
        else:
            rew_end = rew_end + options.GOAL_REW*(options.GAMMA**i) 

    # Reward at Obs
    rew_obs = 0
    for i in range(0, int(total_step*0.5)):
        goal_distance = 60 - i*dt_code*veh_speed 
        if i != int(total_step*0.5)-1:
            rew_obs = rew_obs -options.DIST_MUL*(goal_distance/GOAL_DISTANCE)**2*(options.GAMMA**i) 
        else:
            rew_obs = rew_obs + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 75%
    rew_75 = 0
    for i in range(0, int(total_step*0.75)):
        goal_distance = 60 - i*dt_code*veh_speed 
        if i != int(total_step*0.75)-1:
            rew_75 = rew_75 -options.DIST_MUL*(goal_distance/GOAL_DISTANCE)**2*(options.GAMMA**i) 
        else:
            rew_75 = rew_75 + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 25%
    rew_25 = 0
    for i in range(0, int(total_step*0.25)):
        goal_distance = 60 - i*dt_code*veh_speed 
        if i != int(total_step*0.25)-1:
            rew_25 = rew_25 -options.DIST_MUL*(goal_distance/GOAL_DISTANCE)**2*(options.GAMMA**i) 
        else:
            rew_25 = rew_25 + options.FAIL_REW*(options.GAMMA**i) 

    # EPS Info
    


    ########
    # Print Info
    ########

    print("======================================")
    print("======================================")
    print("        REWARD ESTIMATION")
    print("======================================")
    print("Expected Total Step    : ", total_step)
    print("Expected Reward (25)   : ", rew_25)
    print("Expected Reward (Obs)  : ", rew_obs)
    print("Expected Reward (75)   : ", rew_75)
    print("Expected Reward (Goal) : ", rew_end)
    print("======================================")
    print("        EPS ESTIMATION")
    print("Expected Step per Epi  : ", total_step*0.5)
    print("Total Steps            : ", total_step*0.5*options.MAX_EPISODE)
    print("EPS at Last Episode    : ", options.INIT_EPS*options.EPS_DECAY**(total_step*0.5*options.MAX_EPISODE/options.EPS_ANNEAL_STEPS)  )
    print("======================================")
    print("======================================")


    return











###############################33
# TRAINING
#################################33/

# Initialize to original scene
# Input:
#   veh_index: list of indicies of vehicles
def initScene( veh_index_list, randomize = False):
    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( clientID, vehicle_handle[veh_index], vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
        # Reset position of vehicle. Randomize x-position if enabled
        if randomize == False:
            err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle[veh_index],-1,[veh_index*20-3 + 0,0,0.2],vrep.simx_opmode_blocking)
        else:
            x_pos = random.uniform(-1,-3)
            err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle[veh_index],-1,[veh_index*20 + x_pos,0,0.2],vrep.simx_opmode_blocking)

        # Reset Orientation of vehicle
        err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle[veh_index],-1,[0,0,math.radians(90)],vrep.simx_opmode_blocking)

        # Reset position of motors & steering
        setMotorPosition(clientID, steer_handle[veh_index], 0)

    vrep.simxSynchronousTrigger(clientID);                              # Step one simulation while dynamics disabled to move object

    for veh_index in veh_index_list:
        vrep.simxSetModelProperty( clientID, vehicle_handle[veh_index], 0 , vrep.simx_opmode_blocking   )      # enable dynamics

        # Reset motor speed
        setMotorSpeed(clientID, motor_handle[veh_index], options.INIT_SPD)

        # Set initial speed of vehicle
        vrep.simxSetObjectFloatParameter(clientID, vehicle_handle[veh_index], vrep.sim_shapefloatparam_init_velocity_y, options.INIT_SPD/3.6, vrep.simx_opmode_blocking)
       
        # Read sensor
        dState, dDistance = readSensor(clientID, sensor_handle[veh_index], vrep.simx_opmode_buffer)         # try it once for initialization

        # Reset position of obstacle
        if randomize == True:
            if random.random() > 0.5:
                err_code = vrep.simxSetObjectPosition(clientID,obs_handle[veh_index],-1,[veh_index*20 + -3.3,30,1.1],vrep.simx_opmode_blocking)
            else:
                err_code = vrep.simxSetObjectPosition(clientID,obs_handle[veh_index],-1,[veh_index*20 + 1.675,30,1.1],vrep.simx_opmode_blocking)
        
        # Reset position of dummy    
        if randomize == False:
            pass
        else:
            pass
    #        x_pos = random.uniform(-1,-7.25)
    #        err_code = vrep.simxSetObjectPosition(clientID,dummy_handle,-1,[x_pos,60,0.2],vrep.simx_opmode_blocking)



########################
# MAIN
########################
if __name__ == "__main__":
    options = get_options()
    print(str(options).replace(" ",'\n'))

    # Set Seed
    np.random.seed(1)
    random.seed(1)
    tf.set_random_seed(1)

    ######################################33
    # SET 'GLOBAL' Variables
    ######################################33
    START_TIME       = datetime.datetime.now() 
    START_TIME_STR   = str(START_TIME).replace(" ","_")

    # Save options
    if not os.path.exists("./checkpoints-vehicle"):
        os.makedirs("./checkpoints-vehicle")
    option_file = open("./checkpoints-vehicle/options_"+START_TIME_STR+'.txt', "w")
    option_file.write(
        re.sub(
            r', ',
            r'\n',
            str(options)
        )
    )
    option_file.close()

    # Print Rewards
    printRewards()

    # Get client ID
    vrep.simxFinish(-1) 
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID == -1:
        print("ERROR: Cannot establish connection to vrep.")
        sys.exit()

    # Set Sampling time
    vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, 0.025, vrep.simx_opmode_oneshot)

    # start simulation
    vrep.simxSynchronous(clientID,True)

    #####################
    # GET HANDLES
    #####################

    print("Getting handles...")
    motor_handle, steer_handle = getMotorHandles()
    sensor_handle = getSensorHandles()

    # Get vehicle handle
    vehicle_handle = np.zeros(options.VEH_COUNT, dtype=int)
    for i in range(0,options.VEH_COUNT):
        err_code, vehicle_handle[i] = vrep.simxGetObjectHandle(clientID, "dyros_vehicle" + str(i), vrep.simx_opmode_blocking)

    # Get goal point handle
    dummy_handle = np.zeros(options.VEH_COUNT, dtype=int)
    for i in range(0,options.VEH_COUNT):
        err_code, dummy_handle[i] = vrep.simxGetObjectHandle(clientID, "GoalPoint" + str(i), vrep.simx_opmode_blocking)

    # Get obstacle handle
    obs_handle = np.zeros(options.VEH_COUNT, dtype=int)
    for i in range(0,options.VEH_COUNT):
        err_code, obs_handle[i] = vrep.simxGetObjectHandle(clientID, "obstacle" + str(i), vrep.simx_opmode_blocking)

    # Make Large handle list for communication
    handle_list = [options.VEH_COUNT, SENSOR_COUNT] 
    # Make handles into big list
    for v in range(0,options.VEH_COUNT):
        handle_list = handle_list + [vehicle_handle[v]] + sensor_handle[v].tolist() + [dummy_handle[v]]

    ########################
    # Initialize Test Scene
    ########################

    # Data
    sensorData = np.zeros(SENSOR_COUNT)   
    sensorDetection = np.zeros(SENSOR_COUNT)   
    #vehPosData = []

    ##############
    # TF Setup
    ##############
    # Define placeholders to catch inputs and add options
    options         = get_options()
    agent_train     = QAgent(options,'Training')
    agent_target    = QAgent(options,'Target')
    sess            = tf.InteractiveSession()
    

#    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#    sess = tf.Session()
   
    rwd                  = tf.placeholder(tf.float32, [None, ], name='reward')

    # Copying Variables (taken from https://github.com/akaraspt/tiny-dqn-tensorflow/blob/master/main.py)
    target_vars = agent_target.getTrainableVarByName()
    online_vars = agent_train.getTrainableVarByName()

    copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops)
        
    sess.run(tf.global_variables_initializer())
    copy_online_to_target.run()         # Copy init weights

    # saving and loading networks
    if options.NO_SAVE == False:
        saver = tf.train.Saver( max_to_keep = 20 )
        checkpoint = tf.train.get_checkpoint_state("checkpoints-vehicle")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("=================================================")
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            print("=================================================")
        else:
            print("=================================================")
            print("Could not find old network weights")
            print("=================================================")

    # Some initial local variables
    feed = {}
    eps = options.INIT_EPS
    global_step = 0

    # The replay memory.
    if options.disable_PER == False:
        replay_memory = Memory(options.MAX_EXPERIENCE, absolute_error_upperbound=2000)
    else:
        replay_memory = Memory(options.MAX_EXPERIENCE, disable_PER = True)
        
    ########################
    # END TF SETUP
    ########################

    printTFvars()

    ###########################        
    # DATA VARIABLES
    ###########################        
    reward_data         = np.empty(0)
    avg_loss_value_data = np.empty(1)
#    veh_pos_data        = np.empty([options.MAX_EPISODE, options.MAX_TIMESTEP, 2])
    avg_epi_reward_data = np.zeros(options.MAX_EPISODE)
    epi_reward_data = np.zeros(0)
    track_eps           = []
    track_eps.append((0,eps))

    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)







    ###########################        
    # Initialize Variables
    ###########################        
   
    # Some variables
    reward_stack = np.zeros(options.VEH_COUNT)                              # Holds rewards of current step
    action_stack = np.zeros((options.VEH_COUNT, options.ACTION_DIM))        # action_stack[k] is the array of optinos.ACTION_DIM with one hot encoding
    epi_step_stack = np.zeros(options.VEH_COUNT)      # Count number of step for each vehicle in each episode
    epi_reward_stack = np.zeros(options.VEH_COUNT)                          # Holds reward of current episode
    epi_counter  = 0                                                # Counts # of finished episodes
    epi_done = np.zeros(options.VEH_COUNT) 
    eps_tracker = np.zeros(options.MAX_EPISODE+options.VEH_COUNT+1)
 
    # Initialize Scene

    initScene( list(range(0,options.VEH_COUNT)), randomize = False)               # initialize

    # List of deque to store data
    sensor_queue = []
    goal_queue   = []
    for i in range(0,options.VEH_COUNT):
        sensor_queue.append( deque() )
        goal_queue.append( deque() )


    # initilize them with initial data
    veh_pos, veh_heading, dDistance, gInfo = getVehicleStateLUA()
    for i in range(0,options.VEH_COUNT):
        # Copy initial state FRAME_COUNT*2 times. First FRAME_COUNT will store state of previous, and last FRAME_COUNT store state of current
        for q in range(0,options.FRAME_COUNT*2):
            sensor_queue[i].append(dDistance[i])
            goal_queue[i].append(gInfo[i])

    curr_weight = tf.get_default_graph().get_tensor_by_name('Training/h_s1/kernel:0').eval()
    prev_weight = tf.get_default_graph().get_tensor_by_name('Training/h_s1/kernel:0').eval()
    #curr_weight_target = tf.get_default_graph().get_tensor_by_name('Target/h_s1/kernel:0').eval()
    #prev_weight_target = tf.get_default_graph().get_tensor_by_name('Target/h_s1/kernel:0').eval()
    # Global Step Loop
    while epi_counter <= options.MAX_EPISODE:
        prev_weight = curr_weight
        curr_weight = tf.get_default_graph().get_tensor_by_name('Training/h_s1/kernel:0').eval()
        #print('Training Kernel 0 Diff:', np.linalg.norm(curr_weight - prev_weight))
        #print('WEIGHT:',tf.get_default_graph().get_tensor_by_name('Training/h_s1/kernel:0').eval() )
        #print('BIAS:',tf.get_default_graph().get_tensor_by_name('Training/h_s1/bias:0').eval() )


        #prev_weight_target = curr_weight_target
        #curr_weight_target = tf.get_default_graph().get_tensor_by_name('Target/h_s1/kernel:0').eval()
        #print('Target Kenel 0 Diff:', np.linalg.norm(curr_weight_target - prev_weight_target))

        #GS_START_TIME_STR   = datetime.datetime.now()
        # Decay epsilon
        global_step += options.VEH_COUNT
        if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
            eps = eps * options.EPS_DECAY
            # Save eps for plotting
            #track_eps.append((epi_counter+1,eps)) 

        #print("=====================================")
        #print("Global Step: " + str(global_step) + 'EPS: ' + str(eps) + ' Finished Episodes:' + str(epi_counter) )


        ####
        # Find & Apply Action
        ####
        for v in range(0,options.VEH_COUNT):
            # Get current info to generate input
            observation     = getObs( sensor_queue[v], goal_queue[v], old=False)

            # Get Action. First vehicle always exploit
            if v == 0:
                action_stack[v] = agent_train.sample_action(
                                                    {
                                                        agent_train.observation : np.reshape(observation, (1, -1))
                                                    },
                                                    eps,
                                                    options,
                                                    True
                                                 )
            else:
                action_stack[v] = agent_train.sample_action(
                                                    {
                                                        agent_train.observation : np.reshape(observation, (1, -1))
                                                    },
                                                    eps,
                                                    options
                                                 )
            applySteeringAction( action_stack[v], v )

        ####
        # Step
        ####
        for q in range(0,options.FIX_INPUT_STEP):
            vrep.simxSynchronousTrigger(clientID);

        if options.manual == True:
            input('Press Enter')

        ####
        # Get Next State
        ####
        next_veh_pos, next_veh_heading, next_dDistance, next_gInfo = getVehicleStateLUA()
        #print(next_dDistance)
        for v in range(0,options.VEH_COUNT):
            # Get new Data
            #veh_pos_queue[v].append(next_veh_pos[v])   

            # Update queue
            sensor_queue[v].append(next_dDistance[v])
            sensor_queue[v].popleft()
            goal_queue[v].append(next_gInfo[v])
            goal_queue[v].popleft()

            # Get reward for each vehicle
            reward_stack[v] = -options.DIST_MUL*next_gInfo[v][1]**2        # cost is the distance squared + time it survived

            if v == 0:
                #print(next_dDistance[v])
                #print(next_gInfo[v])
                pass



        #print(next_veh_heading)

        ###
        # Handle Events
        ###
        
        # Reset Done
        epi_done = np.zeros(options.VEH_COUNT)

        # List of resetting vehicles
        reset_veh_list = []

        # Find reset list 
        for v in range(0,options.VEH_COUNT):
            # If vehicle collided, give large negative reward
            collision_detected, collision_sensor = detectCollision(next_dDistance[v])
            if collision_detected == True:
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' collided! Detected Sensor : ' + str(collision_sensor) )
                print('-----------------------------------------')
                
                # Print last Q-value 
                observation     = getObs( sensor_queue[v], goal_queue[v], old=False)
                curr_q_value = sess.run([agent_train.output], feed_dict = {agent_train.observation : np.reshape(observation, (1, -1))})
                print(curr_q_value)

                # Add this vehicle to list of vehicles to reset
                reset_veh_list.append(v)

                # Set flag and reward, and save eps
                reward_stack[v] = options.FAIL_REW
                eps_tracker[epi_counter] = eps
                epi_counter += 1

                # Set done
                epi_done[v] = 1
                # If collided, skip checking for goal point
                continue

            # If vehicle is at the goal point, give large positive reward
            if detectReachedGoal(next_veh_pos[v], next_gInfo[v], next_veh_heading[v]):
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' reached goal point')
                print('-----------------------------------------')
                
                # Reset Simulation
                reset_veh_list.append(v)

                # Set flag and reward
                reward_stack[v] = options.GOAL_REW
                eps_tracker[epi_counter] = eps
                epi_counter += 1

                # Set done
                epi_done[v] = 1

            # If over MAXSTEP
            if epi_step_stack[v] > options.MAX_TIMESTEP:
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' over max step')
                print('-----------------------------------------')

                # Reset Simulation
                reset_veh_list.append(v)

                eps_tracker[epi_counter] = eps
                epi_counter += 1
                

        # Detect being stuck
        #   Not coded yet.
        
        # Reset Vehicles    
        initScene( reset_veh_list, randomize = False)               # initialize

        
        ###########
        # START LEARNING
        ###########

        # Add latest information to memory
        for v in range(0,options.VEH_COUNT):
            # Get observation
            observation             = getObs( sensor_queue[v], goal_queue[v], old = True)
            next_observation        = getObs( sensor_queue[v], goal_queue[v], old = False)

            # Add experience. (observation, action in one hot encoding, reward, next observation, done(1/0) )
            experience = observation, action_stack[v], reward_stack[v], next_observation, epi_done[v]
           
            # Save new memory 
            replay_memory.store(experience)

        # Start training
        if global_step >= options.MAX_EXPERIENCE and options.TESTING == False:
            for tf_train_counter in range(0,options.VEH_COUNT):
                # Obtain the mini batch. (Batch Memory is '2D array' with BATCH_SIZE X size(experience)
                tree_idx, batch_memory, ISWeights_mb = replay_memory.sample(options.BATCH_SIZE)

            
                # Get state/action/next state from obtained memory. Size same as queues
                states_mb       = np.array([each[0][0] for each in batch_memory])           # BATCH_SIZE x STATE_DIM 
                actions_mb      = np.array([each[0][1] for each in batch_memory])           # BATCH_SIZE x ACTION_DIM
                rewards_mb      = np.array([each[0][2] for each in batch_memory])           # 1 x BATCH_SIZE
                next_states_mb  = np.array([each[0][3] for each in batch_memory])   
                done_mb         = np.array([each[0][4] for each in batch_memory])   

                # Get Target Q-Value
                feed.clear()
                feed.update({agent_train.observation : next_states_mb})

                # Calculate Target Q-value. Uses double network. First, get action from training network
                action_train = np.argmax( agent_train.output.eval(feed_dict=feed), axis=1 )

                if options.disable_DN == False:
                    feed.clear()
                    feed.update({agent_target.observation : next_states_mb})

                    # Using Target + Double network
                    q_target_val = rewards_mb + options.GAMMA * agent_target.output.eval(feed_dict=feed)[np.arange(0,options.BATCH_SIZE),action_train]
                else:
                    feed.clear()
                    feed.update({agent_target.observation : next_states_mb})

                    # Just using Target Network.
                    q_target_val = rewards_mb + options.GAMMA * np.amax( agent_target.output.eval(feed_dict=feed), axis=1)
           
                # set q_target to reward if episode is done
                for v_mb in range(0,options.BATCH_SIZE):
                    if done_mb[v_mb] == 1:
                        q_target_val[v_mb] = rewards_mb[v_mb]
 


                # Gradient Descent
                feed.clear()
                feed.update({agent_train.observation : states_mb})
                feed.update({agent_train.act : actions_mb})
                feed.update({agent_train.target_Q : q_target_val } )        # Add target_y to feed
                feed.update({agent_train.ISWeights : ISWeights_mb   })

                with tf.variable_scope("Training"):            
                    step_loss_per_data, step_loss_value, _ = sess.run([agent_train.loss_per_data, agent_train.loss, agent_train.optimizer], feed_dict = feed)
                    #print(rewards_mb)
                    #print(step_loss_value)

                    # Use sum to calculate average loss of this episode.
                    avg_loss_value_data = np.append(avg_loss_value_data,np.mean(step_loss_per_data))
                    #if tf_train_counter == 0:
                        #print(step_loss_per_data)
        
                # Update priority
                replay_memory.batch_update(tree_idx, step_loss_per_data)

        ###############
        # Miscellaneous
        ###############
        # Update Target
        if global_step % options.TARGET_UPDATE_STEP == 0:
            print('-----------------------------------------')
            print("Updating Target network.")
            print('-----------------------------------------')
            copy_online_to_target.run()

        # Update rewards
        for v in range(0,options.VEH_COUNT):
            epi_reward_stack[v] = epi_reward_stack[v] + reward_stack[v]*(options.GAMMA**epi_step_stack[v])

        # Print Rewards
        for v in reset_veh_list:
            print('========')
            print('Vehicle #:', v)
            print('\tGlobal Step:' + str(global_step))
            print('\tEPS: ' + str(eps))
            print('\tEpisode #: ' + str(epi_counter) + ' Step: ' + str(epi_step_stack[v]))
            print('\tEpisode Reward: ' + str(epi_reward_stack[v])) 
            print('Last Loss: ',avg_loss_value_data[-1])
            print('========')
            print('')

        # Update Counters
        epi_step_stack = epi_step_stack + 1

        # Reset rewards for finished vehicles
        for v in reset_veh_list:
            epi_reward_data = np.append(epi_reward_data,epi_reward_stack[v]) 
            if v == 0:
                reward_data = np.append(reward_data,epi_reward_stack[0])

            epi_reward_stack[v] = 0
            epi_step_stack[v] = 0

        #GS_END_TIME   = datetime.datetime.now()
        #print(GS_END_TIME - GS_START_TIME_STR)

        # Stop and Restart Simulation Every X episodes
        if global_step % options.RESET_STEP == 0:
            print('-----------------------------------------')
            print("Resetting...")
            print('-----------------------------------------')
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
            
            # Wait until simulation is stopped.
            simulation_status = 1
            bit_mask = 1
            #while bit_mask & simulation_status != 0:       # Get right-most bit and check if it is 1
                #ret, simulation_status = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
                #print(simulation_status)
                #print('Ret:' + str(ret))
                #print(bit_mask & simulation_status)
                #print("{0:b}".format(simulation_status))
                #time.sleep(0.1)
            time.sleep(9)

            # Start simulation and initilize scene
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
            vrep.simxSynchronous(clientID,True)
            time.sleep(1.0)
            initScene( list(range(0,options.VEH_COUNT)), randomize = False)               # initialize
        
        # save progress every 1000 episodes AND testing is disabled
        if options.TESTING == False:
            if global_step // options.SAVER_RATE >= 1 and global_step % options.SAVER_RATE == 0 and options.NO_SAVE == False:
                print('-----------------------------------------')
                print("Saving network...")
                print('-----------------------------------------')
                saver.save(sess, 'checkpoints-vehicle/vehicle-dqn_s' + START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step))
                #print("Done") 

                print('-----------------------------------------')
                print("Saving data...") 
                print('-----------------------------------------')
                # Save Reward Data
                outfile = open( 'result_data/reward_data/reward_data_' + START_TIME_STR, 'wb')  
                pickle.dump( epi_reward_data, outfile )
                outfile.close()

                # Save vehicle position
#                outfile = open( 'result_data/veh_data/veh_pos_data_' + START_TIME_STR, 'wb')  
#                pickle.dump( veh_pos_data, outfile )
#                outfile.close()

                # Save loss data
                outfile = open( 'result_data/loss_data/avg_loss_value_data_' + START_TIME_STR, 'wb')  
                pickle.dump( avg_loss_value_data, outfile )
                outfile.close()
                #print("Done") 

    # stop the simulation & close connection
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)

    ######################################################
    # Post Processing
    ######################################################

    # Print Current Time
    END_TIME = datetime.datetime.now()
    print("===============================")
    print("Start Time: ",START_TIME_STR)
    print("End Time  : ",str(END_TIME).replace(" ","_"))
    print("Duration  : ",END_TIME-START_TIME)
    print("===============================")

    #############################3
    # Visualize Data
    ##############################3


    # Plot Episode reward
    plt.figure(0)
    fig, ax1 = plt.subplots()
    ax1.plot(epi_reward_data)

    # Plot eps value
    ax2 = ax1.twinx()
    ax2.plot(eps_tracker,linestyle='--', color='red')

    ax1.set_title("Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    fig.savefig('result_data/reward_data/reward_data_' + START_TIME_STR + '.png') 


    # Plot Episode running average of rewards
    plt.figure(1)
    fig, ax1 = plt.subplots()
    ax1.plot(runningAverage(epi_reward_data,options.RUNNING_AVG_STEP))

    # Plot eps value
    #ax2 = ax1.twinx()
    #ax2.plot(eps_tracker,linestyle='--', color='red')

    ax1.set_title("Running Average of Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward (Running Average)")
    fig.savefig('result_data/reward_data/reward_data_' + START_TIME_STR + '_average.png') 

    # Plot Average Step Loss
    plt.figure(2)
    fig, ax2 = plt.subplots()
    ax2.plot(avg_loss_value_data)
    
    ax2.set_title("Average Loss per Batch Step")
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Avg Loss")
    fig.savefig('result_data/avg_loss_value_data/avg_loss_value_data_' + START_TIME_STR + '.png') 

    # Plot Greedy Reward
    plt.figure(3)
    fig, ax3 = plt.subplots()
    ax3.plot(reward_data)
    ax3.set_title("Cumulative Reward for Greedy Action")
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Cumulative Reward")
    fig.savefig('result_data/reward_data/greedy_reward_data_' + START_TIME_STR + '.png') 

    # END
    sys.exit()

    # Plot sensor data
#    t = range(0,220)
    plt.figure(0)
#    for i in range(0,SENSOR_COUNT):
#        plt.plot(sensorData[1:None,i]*sensorDetection[1:None,i], label="Sensor " + str(i) )       # plot sensor0
    plt.legend()
    plt.title("Sensor Data")
    plt.xlabel("Time Step")
    plt.ylabel("Distance")
#    plt.show()



    # Alert User
    mstr='RL Simulation Done!'
    os.system('notify-send '+mstr)


