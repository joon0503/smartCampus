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

MAX_DISTANCE = 15
GOAL_DISTANCE = 60
SENSOR_COUNT = 9

def get_options():
    parser = ArgumentParser(
        description='File for learning'
        )
    parser.add_argument('--MAX_EPISODE', type=int, default=25001,
                        help='max number of episodes iteration\n')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=100,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=11,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.95,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=1.0,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-1,
                        help='finial probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.995,
                        help='epsilon decay rate')
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=2000,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=2.5e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=10000,
                        help='size of experience replay memory')
    parser.add_argument('--SAVER_RATE', type=int, default=1000,
                        help='Save network after this number of episodes')
    parser.add_argument('--FIX_INPUT_STEP', type=int, default=2,
                        help='Fix chosen input for this number of steps')
    parser.add_argument('--TARGET_UPDATE_STEP', type=int, default=100,
                        help='Number of steps required for target update')
    parser.add_argument('--BATCH_SIZE', type=int, default=64,
                        help='mini batch size'),
    parser.add_argument('--H1_SIZE', type=int, default=80,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=80,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=40,
                        help='size of hidden layer 3')
    parser.add_argument('--RESET_EPISODE', type=int, default=250,
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
                        help='Disable the usage of double network.')
    parser.add_argument('--disable_duel', action='store_true',
                        help='Disable the usage of double network.')
    parser.add_argument('--FRAME_COUNT', type=int, default=1,
                        help='Number of frames to be used')
    parser.add_argument('--ACT_FUNC', type=str, default='elu',
                        help='Activation function')
    options = parser.parse_args()
    return options


# Class for Neural Network
class QAgent:
    # A naive neural network with 3 hidden layers and relu as non-linear function.
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
                                             activation = act_function,
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

                self.adv_est    = tf.layers.dense( inputs=self.h_layer_val,
                                             units=options.ACTION_DIM,
                                             activation = None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="adv_est"
                                           )
             
                self.output = self.val_est + tf.subtract(self.adv_est, tf.reduce_mean(self.adv_est, axis=1, keepdims=True) ) 
            ######################################:
            ## END Constructing Neural Network
            ######################################:

            # Prediction of given specific action
            self.Q = tf.reduce_sum( tf.multiply(self.output, self.act), axis=1)

            # Absolute Loss
            self.loss_per_data = tf.abs(self.target_Q - self.Q, name='loss_per_data')    
            
            # Loss for Optimization
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_Q, self.Q))
            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)


    # Sample action with random rate eps
    def sample_action(self, feed, eps, options):
        if random.random() <= eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = random.randrange(options.ACTION_DIM)
#            action = random.uniform(0,1)
        else:
            act_values = self.output.eval(feed_dict=feed)
            if options.TESTING == True:
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

def readSensor( clientID, sensorHandles, op_mode=vrep.simx_opmode_streaming ):
    dState      = []
    dDistance   = []
    for sHandle in sensorHandles:
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
def detectCollision(dDistance, dState):
    for i in range(SENSOR_COUNT):
        if dDistance[i] < 1.1/MAX_DISTANCE and dState[i] == True:
            return True, i
    
    return False, -1    

# Get goal point vector. Returns angle and distance
# Output
# goal_angle: angle to the goal point in radians / pi
#             Positive angle mean goal point is on the right of vehicle, negative mean goal point is on the left
# goal_distance: distance to the goal point / MAX_DISTANCE
def getGoalPoint():
    _, vehPos = vrep.simxGetObjectPosition( clientID, vehicle_handle, -1, vrep.simx_opmode_blocking)            
    _, dummyPos = vrep.simxGetObjectPosition( clientID, dummy_handle, -1, vrep.simx_opmode_blocking)            

    # Calculate the distance
    delta_distance = np.array(dummyPos) - np.array(vehPos)              # delta x, delta y
    goal_distance = np.linalg.norm(delta_distance)

    # calculate angle
    goal_angle = math.atan( delta_distance[0]/delta_distance[1] )       # delta x / delta y
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
# Output: 4 list of float
#  first list    : Sensor distance
#  second list   : Sensor detection state (0-False, 1-True) 
#  third list    : [goal angle, goal distance]
#  fourth list   : vehicle position (x,y)
#  fifth list    : vehicle heading from -1 to 1. 1 if facing left, -1 if facing right, 0 if facing front
def getVehicleState():
    # Read sensor
    dState, dDistance   = readSensor(clientID, sensor_handle)

    # Read Vehciel Position
    _, vehPos           = vrep.simxGetObjectPosition( clientID, vehicle_handle, -1, vrep.simx_opmode_blocking)            

    # Read Goal Point Angle & Distance
    gAngle, gDistance   = getGoalPoint()

    # Read vehicle heading
    _, vehEuler         = vrep.simxGetObjectOrientation( clientID, vehicle_handle, -1, vrep.simx_opmode_blocking)            

    # vehEuler - [alpha, beta, gamma] in radians. When vehicle is facing goal point, we have gamma = +90deg = pi/2. Facing right - +0deg, Facing left - +180deg Translate such that it is 0, positive for left, negative for right and within -1 to 1
    vehHeading          = (vehEuler[2]-math.pi*0.5)/(math.pi/2)

    return dDistance, dState, [gAngle, gDistance], vehPos, vehHeading


# Given one-hot-encoded action array of steering angle, apply it to the vehicle
# Input:
#   action: list of 1/0s. 1 means this action is applied
def applySteeringAction(action):
    # Define maximum/minimum steering in degrees
    MAX_STEER = 15
    MIN_STEER = -15

    # Delta of angle between each action
    action_delta = (MAX_STEER - MIN_STEER) / (options.ACTION_DIM-1)

    # Calculate desired angle
    desired_angle = MAX_STEER - np.argmax(action) * action_delta

    # Set steering position
    setMotorPosition(clientID, steer_handle, desired_angle)
    
    return


# Get Motor/Sensor Handles
# Input: clientID?
def getMotorHandles():
    # Get Motor Handles
    _,h1  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorLeft", vrep.simx_opmode_blocking)
    _,h2  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorRight", vrep.simx_opmode_blocking)
    _,h3  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisRight", vrep.simx_opmode_blocking)
    _,h4  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisLeft", vrep.simx_opmode_blocking)
    _,h5  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringLeft", vrep.simx_opmode_blocking)
    _,h6  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringRight", vrep.simx_opmode_blocking)

    motor_handle = [h1, h2]
    steer_handle = [h5, h6]

    return motor_handle, steer_handle

def getSensorHandles():
    # Get Sensor Handles
    sensor_handle = []
    for i in range(0,SENSOR_COUNT):
        _,temp_handle  = vrep.simxGetObjectHandle(clientID, "Proximity_sensor" + str(i), vrep.simx_opmode_blocking)
        sensor_handle.append(temp_handle)

    return sensor_handle

##################################
# TENSORFLOW HELPER
##################################

def printTFvars():
    tvars = tf.trainable_variables()

    for var in tvars:
        print(var)
    return

##################################
# GENERAL
##################################

# From sensor and goal queue, get observation.
# observation is a row vector with all frames of information concatentated to each other
# Recall that queue stores 2*FRAME_COUNT of information. 
# first = True: get oldest info
# first = False: get latest info
def getObs( sensor_queue, goal_queue, first=True):

    if first == True:
        sensor_stack    = np.concatenate(sensor_queue)[0:SENSOR_COUNT*options.FRAME_COUNT]
        goal_stack      = np.concatenate(goal_queue)[0:2*options.FRAME_COUNT]
        observation     = np.concatenate((sensor_stack, goal_stack))    
    else:
        sensor_stack    = np.concatenate(sensor_queue)[SENSOR_COUNT*options.FRAME_COUNT:]
        goal_stack      = np.concatenate(goal_queue)[2*options.FRAME_COUNT:]
        observation     = np.concatenate((sensor_stack, goal_stack))    

    return observation








###############################33
# TRAINING
#################################33/

# Initialize to original scene
def initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = False):
    vrep.simxSetModelProperty( clientID, vehicle_handle, vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
    # Reset position of vehicle. Randomize x-position if enabled
    if randomize == False:
        err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[0,0,0.2],vrep.simx_opmode_blocking)
    else:
        x_pos = random.uniform(-1,-7.25)
        err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[x_pos,0,0.2],vrep.simx_opmode_blocking)

    # Reset Orientation of vehicle
    err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle,-1,[0,0,math.radians(90)],vrep.simx_opmode_blocking)

    # Reset position of motors & steering
    setMotorPosition(clientID, steer_handle, 0)

    vrep.simxSynchronousTrigger(clientID);                              # Step one simulation while dynamics disabled to move object

    vrep.simxSetModelProperty( clientID, vehicle_handle, 0 , vrep.simx_opmode_blocking   )      # enable dynamics

    # Reset motor speed
#    print("Setting motor speed")
    init_speed = 60 # km/hr
    setMotorSpeed(clientID, motor_handle, init_speed)

    # Set initial speed of vehicle
    vrep.simxSetObjectFloatParameter(clientID, vehicle_handle, vrep.sim_shapefloatparam_init_velocity_y, init_speed*(1000/3600), vrep.simx_opmode_blocking)
   
    # Read sensor
    dState, dDistance = readSensor(clientID, sensor_handle, vrep.simx_opmode_buffer)         # try it once for initialization

    # Reset position of obstacle
    if randomize == True:
        if random.random() > 0.5:
            err_code = vrep.simxSetObjectPosition(clientID,obs_handle,-1,[-1,30,1.1],vrep.simx_opmode_blocking)
        else:
            err_code = vrep.simxSetObjectPosition(clientID,obs_handle,-1,[-6.5,30,1.1],vrep.simx_opmode_blocking)
    
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
    print(options)
 
    ######################################33
    # SET 'GLOBAL' Variables
    ######################################33
    START_TIME   = str(datetime.datetime.now()).replace(" ","_")

    # Save options
    if not os.path.exists("./checkpoints-vehicle"):
        os.makedirs("./checkpoints-vehicle")
    option_file = open("./checkpoints-vehicle/options_"+START_TIME+'.txt', "w")
    option_file.write(
        re.sub(
            r', ',
            r'\n',
            str(options)
        )
    )
    option_file.close()

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
    err_code, vehicle_handle = vrep.simxGetObjectHandle(clientID, "dyros_vehicle", vrep.simx_opmode_blocking)

    # Get goal point handle
    err_code, dummy_handle = vrep.simxGetObjectHandle(clientID, "Dummy", vrep.simx_opmode_blocking)

    # Get obstacle handle
    err_code, obs_handle = vrep.simxGetObjectHandle(clientID, "obstacle", vrep.simx_opmode_blocking)

    ########################
    # Initialize Test Scene
    ########################

    # Data
    sensorData = np.zeros(SENSOR_COUNT)   
    sensorDetection = np.zeros(SENSOR_COUNT)   
    vehPosData = []

    ##############
    # TF Setup
    ##############
    # Define placeholders to catch inputs and add options
    options         = get_options()
    agent_train     = QAgent(options,'Training')
    agent_target    = QAgent(options,'Target')
    sess            = tf.InteractiveSession()

#    printTFvars()

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

    ###########################        
    # DATA VARIABLES
    ###########################        
    reward_data         = np.empty(options.MAX_EPISODE)
    avg_loss_value_data = np.empty(options.MAX_EPISODE)
#    veh_pos_data        = np.empty([options.MAX_EPISODE, options.MAX_TIMESTEP, 2])
    avg_epi_reward_data = np.zeros(options.MAX_EPISODE)
    track_eps           = []
    track_eps.append((0,eps))

    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = True)               # initialize

    # EPISODE LOOP
    for j in range(0,options.MAX_EPISODE): 
#        EPI_START_TIME = time.time()
        print("Episode: " + str(j) + ". Global Step: " + str(global_step) + " eps: " + str(eps))

        sum_loss_value = 0
        episode_reward = 0
        vehPosDataTrial = np.empty([options.MAX_TIMESTEP,2])        # Initialize data
        done = 0

        # get data
        dDistance, dState, gInfo, prev_vehPos, prev_Heading = getVehicleState()

        # Initilize queue for storing states.
        # Queue stores latest info at the right, and oldest at the left
        sensor_queue = deque()
        goal_queue = deque()

        # Copy initial state FRAME_COUNT*2 times. First FRAME_COUNT will store state of previous, and last FRAME_COUNT store state of current
        for q in range(0,options.FRAME_COUNT*2):
            sensor_queue.append(dDistance)
            goal_queue.append(gInfo)

    
        for i in range(0,options.MAX_TIMESTEP):     # Time Step Loop
            # Decay epsilon
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY
                # Save eps for plotting
                track_eps.append((j+1,eps)) 


            # Save data
#            sensorData = np.vstack( (sensorData,dDistance) )
#            sensorDetection = np.vstack( (sensorDetection,dState) )
#            vehPosDataTrial[i] = prev_vehPos[0:2]

            # observation
            observation     = getObs( sensor_queue, goal_queue, first=True)

            # Update memory
            action = agent_train.sample_action(
                                                {
                                                    agent_train.observation : np.reshape(observation, (1, -1))
#                                                    agent_train.sensor_input : np.stack(sensor_queue, axis=1),
#                                                    agent_train.goal_input   : np.stack(goal_queue, axis=1)
                                                },
                                                eps,
                                                options
                                             )

            # Apply action
            applySteeringAction( action )
            if options.manual == True:
                input('Press any key to step forward. Curr i:' + str(i))



            # Step simulation by one step
            veh_pos_queue = deque()
            for q in range(0,options.FIX_INPUT_STEP):
                vrep.simxSynchronousTrigger(clientID);

                # Get new Data
                dDistance, dState, gInfo, curr_vehPos, curr_Heading = getVehicleState()
                veh_pos_queue.append(curr_vehPos)   
 
                # Update queue
                sensor_queue.append(dDistance)
                sensor_queue.popleft()
                goal_queue.append(gInfo)
                goal_queue.popleft()

#                next_observation = dDistance + gInfo
                reward = -10*gInfo[1]**2        # cost is the distance squared + time it survived
                
                # If vehicle collided during stepping
                if detectCollision(dDistance,dState)[0] == True:
                    reward = -1e3
                    break

            # If vehicle is stuck somehow
            prev_vehPos = veh_pos_queue[0]
            curr_vehPos = veh_pos_queue[-1]

            if abs(np.asarray(prev_vehPos[0:1]) - np.asarray(curr_vehPos[0:1])) < 0.0005 and i >= 15 and curr_vehPos[1] < 35:
                print('Vehicle Stuck!')
                # Reset Simulation
                initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = True)               # initialize
                
                done = 1

            # If vehicle collided, give large negative reward
            if detectCollision(dDistance,dState)[0] == True:
                print('Vehicle collided!')
                
                # Print last Q-value 
                curr_q_value = sess.run([agent_train.output], feed_dict = {agent_train.observation : np.reshape(observation, (1, -1))})
                print(curr_q_value)

                # Reset Simulation
                initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = True)               # initialize

                # Set flag and reward
                done = 1
                reward = -1e3

            # If vehicle is at the goal point, give large positive reward
            if detectReachedGoal(curr_vehPos, gInfo, curr_Heading):
                print('Reached goal point')
                
                # Reset Simulation
                initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = True)               # initialize

                # Set flag and reward
                done = 1
                reward = 1e3

            # Save new memory
            next_observation = getObs(sensor_queue,goal_queue,first=False)      # Get latest info
            experience = observation, action, reward, next_observation
            replay_memory.store(experience)

            # Record reward
            episode_reward = episode_reward + reward*(options.GAMMA**i)

            if global_step >= options.MAX_EXPERIENCE and options.TESTING == False:
                # Obtain the mini batch
                tree_idx, batch_memory, ISWeights_mb = replay_memory.sample(options.BATCH_SIZE)

                # Get state/action/next state from obtained memory. Size same as queues
                states_mb       = np.array([each[0][0] for each in batch_memory])           # BATCH_SIZE x STATE_DIM 
                actions_mb      = np.array([each[0][1] for each in batch_memory])           # BATCH_SIZE x ACTION_DIM
                rewards_mb      = np.array([each[0][2] for each in batch_memory])           # 1 x BATCH_SIZE
                next_states_mb  = np.array([each[0][3] for each in batch_memory])   

                # Get Target Q-Value
                feed.clear()
                feed.update({agent_train.observation : next_states_mb})
            #    feed.update({next_obs : next_states_mb})

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
                    # Just using Target Network
                    q_target_val = rewards_mb + options.GAMMA * np.amax( agent.target.output.eval(feed_dict=feed), axis=1)
    

                # Gradient Descent
                feed.clear()
                feed.update({agent_train.observation : states_mb})
                feed.update({agent_train.act : actions_mb})
                feed.update({agent_train.target_Q : q_target_val } )        # Add target_y to feed
                feed.update({agent_train.ISWeights : ISWeights_mb   })

                with tf.variable_scope("Training"):            
                    step_loss_per_data, step_loss_value, _ = sess.run([agent_train.loss_per_data, agent_train.loss, agent_train.optimizer], feed_dict = feed)

                    # Use sum to calculate average loss of this episode
                    sum_loss_value += np.mean(step_loss_per_data)
    
        
                # Update priority
                replay_memory.batch_update(tree_idx, step_loss_per_data)

                # Visualizing graph     # use tensorboard --logdir=output
#                writer = tf.summary.FileWriter("output", sess.graph)
#                print(sess.run(loss, feed_dict = feed))
#                writer.close()

            # Update Target
            if global_step % options.TARGET_UPDATE_STEP == 0:
                print("Updating Target network.")
                copy_online_to_target.run()

            # If collided or reached goal point, end this episode
            if done == 1:
                print(episode_reward)
        
                reward_data[j]          = episode_reward
                avg_loss_value_data[j]  = sum_loss_value/(i+1)
#                veh_pos_data[j]         = vehPosDataTrial
                break


        # EPISODE ENDED
        print("====== Episode" + str(j) + " ended at Step " + str(i)+ ". sum_loss_value: " + str(sum_loss_value) + " avg_loss_value: " + str(avg_loss_value_data[j]) )
       
        # Update running average of the reward 
        if j >= options.RUNNING_AVG_STEP:
            avg_epi_reward_data[j] = np.mean(reward_data[j-options.RUNNING_AVG_STEP+1:j])  
          
         
        # Stop and Restart Simulation Every X episodes
        if j % options.RESET_EPISODE == 0:
            print("Resetting...")
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
            
            # Wait until simulation is stopped.
            simulation_status = 1
            bit_mask = 1
            while bit_mask & simulation_status != 0:       # Get right-most bit and check if it is 1
                _, simulation_status = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
#                    print(bit_mask & simulation_status)
#                    print("{0:b}".format(simulation_status))
                time.sleep(0.1)

            # Start simulation and initilize scene
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
            vrep.simxSynchronous(clientID,True)
            time.sleep(1.0)
            initScene(vehicle_handle, steer_handle, motor_handle, obs_handle, dummy_handle, randomize = True)               # initialize
        
        # save progress every 1000 episodes AND testing is disabled
        if options.TESTING == False:
            if j // options.SAVER_RATE >= 1 and j % options.SAVER_RATE == 0 and options.NO_SAVE == False:
                print("Saving network...")
                saver.save(sess, 'checkpoints-vehicle/vehicle-dqn_s' + START_TIME + "_e" + str(j) + "_gs" + str(global_step))
                print("Done") 

                print("Saving data...") 
                # Save Reward Data
                outfile = open( 'result_data/reward_data/reward_data_' + START_TIME, 'wb')  
                pickle.dump( reward_data, outfile )
                outfile.close()

                # Save vehicle position
#                outfile = open( 'result_data/veh_data/veh_pos_data_' + START_TIME, 'wb')  
#                pickle.dump( veh_pos_data, outfile )
#                outfile.close()

                # Save loss data
                outfile = open( 'result_data/loss_data/avg_loss_value_data_' + START_TIME, 'wb')  
                pickle.dump( avg_loss_value_data, outfile )
                outfile.close()
                print("Done") 
        # Line Separator
        print('')

    # stop the simulation & close connection
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)

    ######################################################
    # Post Processing
    ######################################################

    # Print Current Time
    END_TIME = str(datetime.datetime.now()).replace(" ","_")
    print("===============================")
    print(END_TIME)
    print("===============================")

    #############################3
    # Visualize Data
    ##############################3


    # Plot Episode reward
    plt.figure(0)
    fig, ax1 = plt.subplots()
    ax1.plot(avg_epi_reward_data)

    # Plot eps value
    ax2 = ax1.twinx()
    track_eps.append((options.MAX_EPISODE,eps))
    for q in range(0,len(track_eps)-1 ): 
        ax2.plot([track_eps[q][0], track_eps[q+1][0]], [track_eps[q][1],track_eps[q][1]], linestyle='--', color='red')

    ax1.set_title("Average episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    fig.savefig('result_data/reward_data/reward_data_' + START_TIME + '.png') 

    # Plot Average Step Loss
    plt.figure(1)
    fig, ax2 = plt.subplots()
    ax2.plot(avg_loss_value_data)
    
    ax2.set_title("Average Loss of an episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Loss")
    fig.savefig('result_data/avg_loss_value_data/avg_loss_value_data_' + START_TIME + '.png') 

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

