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
from argparse import ArgumentParser


MAX_SCORE_QUEUE_SIZE = 100  # number of episode scores to calculate average performance
MAX_DISTANCE = 15
GOAL_DISTANCE = 60

def get_options():
    parser = ArgumentParser(
        description='File for learning'
        )
    parser.add_argument('--MAX_EPISODE', type=int, default=100000,
                        help='max number of episodes iteration\n')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=200,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=7,
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
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=10000,
                        help='size of experience replay memory')
    parser.add_argument('--SAVER_RATE', type=int, default=1000,
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
    options = parser.parse_args()
    return options


# Class for Neural Network
class QAgent:
    # A naive neural network with 3 hidden layers and relu as non-linear function.
    def __init__(self, options, name):
        self.scope = name
        with tf.variable_scope(name):      # Set scope as name
            self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE], 'W1')
            self.b1 = self.bias_variable([options.H1_SIZE], 'b1')
            self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE], 'W2')
            self.b2 = self.bias_variable([options.H2_SIZE], 'b2')
            self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE], 'W3')
            self.b3 = self.bias_variable([options.H3_SIZE], 'b3')
            self.W4 = self.weight_variable([options.H3_SIZE, options.ACTION_DIM], 'W4')
            self.b4 = self.bias_variable([options.ACTION_DIM], 'b4')

#            self.W3_val = self.weight_variable([options.H2_SIZE, options.H3_SIZE], 'W3_val')
#            self.b3_val = self.bias_variable([options.H3_SIZE], 'b3_val')
#            self.W3_adv = self.weight_variable([options.H2_SIZE, options.H3_SIZE], 'W3_adv')
#            self.b3_adv = self.bias_variable([options.H3_SIZE], 'b3_adv')
#            self.W4_val = self.weight_variable([options.H3_SIZE, 1], 'W4_val')
#            self.b4_val = self.bias_variable([1], 'b4_val')
#            self.W4_adv = self.weight_variable([options.H3_SIZE, options.ACTION_DIM], 'W4_adv')
#            self.b4_adv = self.bias_variable([options.ACTION_DIM], 'b4_adv')
   
    # Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    # Tool function to create weight variables
    def weight_variable(self, shape, var_name=None):
        return tf.Variable(self.xavier_initializer(shape), name=var_name)

    # Tool function to create bias variables
    def bias_variable(self, shape, var_name=None):
        return tf.Variable(self.xavier_initializer(shape), name=var_name)

    # Add options to graph
    def add_value_net(self, options):
        with tf.variable_scope(self.scope):
            observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM], name='observation')
            h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1, name='h1')
            h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2, name='h2')
            h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3, name='h3')
            Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
#            h3_val = tf.nn.relu(tf.matmul(h2, self.W3_val) + self.b3_val, name='h3_val')
#            h3_adv = tf.nn.relu(tf.matmul(h2, self.W3_adv) + self.b3_adv, name='h3_adv')

#            value_est = tf.nn.relu(tf.matmul(h3_val, self.W4_val) + self.b4_val, name='value_est')
#            adv_est = tf.nn.relu(tf.matmul(h3_adv, self.W4_adv) + self.b4_adv, name='adv_est')

#            Q = value_est + tf.subtract(adv_est, tf.reduce_mean(adv_est, axis=1, keepdims=True) ) 

#        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
#        Q = tf.matmul(h3, self.W4) + self.b4
        #q = tf.nn.sigmoid(Q)
        return observation, Q

    # Sample action with random rate eps
    def sample_action(self, Q, feed, eps, options):
        if random.random() <= eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = random.randrange(options.ACTION_DIM)
#            action = random.uniform(0,1)
        else:
            act_values = Q.eval(feed_dict=feed)
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
#   desiredSpd  : single number, speed
def setMotorSpeed( clientID, motorHandles, desiredSpd ):
    err_code = []
    for mHandle in motorHandles:
        err_code.append( vrep.simxSetJointTargetVelocity(clientID, mHandle, desiredSpd, vrep.simx_opmode_blocking) )
        
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
# goal_angle: angle to the goal point in radians
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
    goal_angle = 0.5*goal_angle / math.pi                               # result in -1 to 1

    return goal_angle, goal_distance / GOAL_DISTANCE

# Get current state of the vehicle. It is combination of different information
# Output: 4 list of float
#  first list    : Sensor distance
#  second list   : Sensor detection state (0-False, 1-True) 
#  third list    : [goal angle, goal distance]
#  fourth list   : vehicle position (x,y)
def getVehicleState():
    # Read sensor
    dState, dDistance   = readSensor(clientID, sensor_handle)

    # Read Vehciel Position
    _, vehPos           = vrep.simxGetObjectPosition( clientID, vehicle_handle, -1, vrep.simx_opmode_blocking)            

    # Read Goal Point Angle & Distance
    gAngle, gDistance   = getGoalPoint()

    return dDistance, dState, [gAngle, gDistance], vehPos


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

###############################33
# TRAINING
#################################33/

# Initialize to original scene
def initScene(vehicle_handle, steer_handle, motor_handle, randomize = False):
    # Reset position of vehicle. Randomize x-position if enabled
    if randomize == False:
        err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[0,0,0.2],vrep.simx_opmode_blocking)
    else:
        x_pos = (random.random()-2)*1    
        err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[x_pos,0,0.2],vrep.simx_opmode_blocking)

    # Reset Orientation of vehicle
    err_code = vrep.simxSetObjectOrientation(clientID,vehicle_handle,-1,[0,0,math.radians(90)],vrep.simx_opmode_blocking)

    # Reset position of motors & steering
    setMotorPosition(clientID, steer_handle, 0)

    # Reset motor speed
#    print("Setting motor speed")
    setMotorSpeed(clientID, motor_handle, 5)
   
    # Read sensor
    dState, dDistance = readSensor(clientID, sensor_handle, vrep.simx_opmode_buffer)         # try it once for initialization

########################
# MAIN
########################
if __name__ == "__main__":
    options = get_options()
    print(options) 
    ######################################33
    # SET 'GLOBAL' Variables
    ######################################33
    SENSOR_COUNT = 5
    START_TIME   = str(datetime.datetime.now()).replace(" ","_")


    # Get client ID
    vrep.simxFinish(-1) 
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID == -1:
        print("ERROR: Cannot establish connection to vrep.")
        sys.exit()

    # Set Sampling time
    vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, 0.05, vrep.simx_opmode_oneshot)

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

#    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#    sess = tf.Session()
   
    obs, Q_train         = agent_train.add_value_net(options)
    next_obs, Q_target   = agent_target.add_value_net(options)
    act                  = tf.placeholder(tf.float32, [None, options.ACTION_DIM],name='action')
    rwd                  = tf.placeholder(tf.float32, [None, ], name='reward')
    target_y             = tf.placeholder(tf.float32, [None, ], name='target_y' )  
 
    values1 = tf.reduce_sum(tf.multiply(Q_train, act), reduction_indices=1, name='Q_val_Current')          # Q-value of current network

    loss = tf.reduce_mean(tf.square(values1 - target_y), name='loss')                         # loss
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)


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
    exp_pointer = 0
    learning_finished = False

    # The replay memory. But no replay memory for now
    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])

    # END TF SETUP

    ###########################        
    # DATA VARIABLES
    ###########################        
    reward_data         = np.empty(options.MAX_EPISODE)
    avg_loss_value_data = np.empty(options.MAX_EPISODE)
    veh_pos_data        = np.empty([options.MAX_EPISODE, options.MAX_TIMESTEP, 2])
    avg_epi_reward_data = np.zeros(options.MAX_EPISODE)

    # EPISODE LOOP
    for j in range(0,options.MAX_EPISODE): 
#        EPI_START_TIME = time.time()
        print("Episode: " + str(j) + ". Global Step: " + str(global_step) + " eps: " + str(eps))

        sum_loss_value = 0
        episode_reward = 0
        vehPosDataTrial = np.empty([options.MAX_TIMESTEP,2])        # Initialize data
        done = 0

        # Start simulation and initilize scene
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
        setMotorPosition(clientID, steer_handle, 0)
        setMotorSpeed(clientID, motor_handle, 5)

        # Time reward
        time_reward = 0

        for i in range(0,options.MAX_TIMESTEP):     # Time Step Loop
           # if i % 10 == 0:
#            print("\tStep:" + str(i))

            # Decay epsilon
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY

            # get data
            dDistance, dState, gInfo, vehPos = getVehicleState()
            observation = dDistance + gInfo

            # Save data
#            sensorData = np.vstack( (sensorData,dDistance) )
#            sensorDetection = np.vstack( (sensorDetection,dState) )
            vehPosDataTrial[i] = vehPos[0:2]

            # Update memory
            obs_queue[exp_pointer] = observation
            action = agent_train.sample_action(Q_train, {obs : np.reshape(observation, (1, -1))}, eps, options)
            act_queue[exp_pointer] = action

            # Print variables
#            print(agent_train.b1.eval())


            # Apply action
            applySteeringAction( action )
            if options.manual == True:
                input('Press any key to step forward.')

            # Step simulation by one step
            for q in range(0,options.FIX_INPUT_STEP):
                vrep.simxSynchronousTrigger(clientID);

            # Get new Data
            dDistance, dState, gInfo, vehPos = getVehicleState()
            observation = dDistance + gInfo
            reward = -10*gInfo[1]**2 + time_reward         # cost is the distance squared + time it survived

            # If vehicle collided, give large negative reward
            if detectCollision(dDistance,dState)[0] == True:
                print('Vehicle collided!')

                # Reset Simulation
                vrep.simxSetModelProperty( clientID, vehicle_handle, vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
                initScene(vehicle_handle, steer_handle, motor_handle, randomize = True)               # initialize
                vrep.simxSynchronousTrigger(clientID);                              # Step one simulation while dynamics disabled to move object
                vrep.simxSetModelProperty( clientID, vehicle_handle, 0 , vrep.simx_opmode_blocking   )      # enable dynamics

                # Set flag and reward
                done = 1
                reward = -1e3

            # If vehicle is at the goal point, give large positive reward
            if abs( gInfo[1] ) < 0.5/MAX_DISTANCE:
                print('Reached goal point')
                
                # Reset Simulation
                vrep.simxSetModelProperty( clientID, vehicle_handle, vrep.sim_modelproperty_not_dynamic , vrep.simx_opmode_blocking   )         # Disable dynamic
                initScene(vehicle_handle, steer_handle, motor_handle, randomize = True)               # initialize
                vrep.simxSynchronousTrigger(clientID);                              # Step one simulation while dynamics disabled to move object
                vrep.simxSetModelProperty( clientID, vehicle_handle, 0 , vrep.simx_opmode_blocking   )      # enable dynamics

                # Set flag and reward
                done = 1
                reward = 1e4

            # Record reward
            episode_reward = episode_reward + reward*(options.GAMMA**i)

            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation
    
            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0 # Refill the replay memory if it is full
  
            if global_step >= options.MAX_EXPERIENCE and options.TESTING == False:
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)

                # Get Target Q-Value
                feed.update({next_obs : next_obs_queue[rand_indexs]})
                # Calculate Target Q-value
                q_target_val = rwd_queue[rand_indexs] + options.GAMMA * np.amax( Q_target.eval(feed_dict=feed), axis=1)

                # Gradient Descent
                feed.clear()
                feed.update({obs : obs_queue[rand_indexs]})
                feed.update({act : act_queue[rand_indexs]})
                feed.update({rwd : rwd_queue[rand_indexs]})
                feed.update({next_obs : next_obs_queue[rand_indexs]})
                feed.update( {target_y :q_target_val } )        # Add target_y to feed

                b_before = agent_train.b1.eval()
                with tf.variable_scope("Training"):            
                    step_loss_value, _ = sess.run([loss, train_step], feed_dict = feed)

                    # Use sum to calculate average loss of this episode
                    sum_loss_value += step_loss_value
    
                b_after = agent_train.b1.eval()
                delta = abs(np.mean(b_before-b_after))
                if delta < 1e-10:
                    print("WARNING: Update too small!" + str( delta ))

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
                veh_pos_data[j]         = vehPosDataTrial
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
            time.sleep(1.0)
            setMotorPosition(clientID, steer_handle, 0)
            setMotorSpeed(clientID, motor_handle, 5)
        
        # save progress every 1000 episodes AND testing is disabled
        if options.TESTING == False:
            if j // options.SAVER_RATE >= 1 and j % options.SAVER_RATE == 0 and options.NO_SAVE == False:
                print("Saving network...")
                saver.save(sess, 'checkpoints-vehicle/vehicle-dqn_s' + START_TIME + "_e" + str(j) + "_gs" + str(global_step))
                print("Done") 

                print("Saving data...") 
                # Save Reward Data
                outfile = open( 'result_data/reward_data/reward_data_' + START_TIME + " ", 'wb')  
                pickle.dump( reward_data, outfile )
                outfile.close()

                # Save vehicle position
                outfile = open( 'result_data/veh_data/veh_pos_data_' + START_TIME + " ", 'wb')  
                pickle.dump( veh_pos_data, outfile )
                outfile.close()

                # Save loss data
                outfile = open( 'result_data/loss_data/avg_loss_value_data_' + START_TIME + " ", 'wb')  
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
    plt.plot(avg_epi_reward_data)
    plt.title("Average episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    # Plot Average Step Loss
    plt.figure(0)
    plt.plot(avg_loss_value_data)
    plt.title("Average Loss of an episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Loss")
    plt.show()
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

