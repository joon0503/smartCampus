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
from argparse import ArgumentParser


OUT_DIR = 'cartpole-experiment' # default saving directory
MAX_SCORE_QUEUE_SIZE = 100  # number of episode scores to calculate average performance
SENSOR_COUNT = 11

def get_options():
    parser = ArgumentParser()
    parser.add_argument('--MAX_EPISODE', type=int, default=10,
                        help='max number of episodes iteration')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=1000,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=24,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.9,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=1.0,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-5,
                        help='finial probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.95,
                        help='epsilon decay rate')
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=100,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=100,
                        help='size of experience replay memory')
    parser.add_argument('--BATCH_SIZE', type=int, default=24,
                        help='mini batch size'),
    parser.add_argument('--H1_SIZE', type=int, default=256,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=256,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=256,
                        help='size of hidden layer 3')
    parser.add_argument('--manual','-m', action='store_true',
                        help='Step simulation manually')
    parser.add_argument('--USE_SAVE','-us', action='store_true',
                        help='Use saved tensorflow network')
    options = parser.parse_args()
    return options


# Class for Neural Network
class QAgent:
    
    # A naive neural network with 3 hidden layers and relu as non-linear function.
    def __init__(self, options):
        self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE])
        self.b1 = self.bias_variable([options.H1_SIZE])
        self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
        self.b2 = self.bias_variable([options.H2_SIZE])
        self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
        self.b3 = self.bias_variable([options.H3_SIZE])
        self.W4 = self.weight_variable([options.H3_SIZE, options.ACTION_DIM])
        self.b4 = self.bias_variable([options.ACTION_DIM])
    
    # Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    # Tool function to create weight variables
    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # Tool function to create bias variables
    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # Add options to graph
    def add_value_net(self, options):
        observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM])
        h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
        #q = tf.nn.sigmoid(Q)
        return observation, Q

    # Sample action with random rate eps
    def sample_action(self, Q, feed, eps, options):
        act_values = Q.eval(feed_dict=feed)
        if random.random() <= eps:
            # pick random action
            action_index = random.randrange(options.ACTION_DIM)
#            action = random.uniform(0,1)
        else:
            action_index = np.argmax(act_values)
        action = np.zeros(options.ACTION_DIM)
        action[action_index] = 1
        return action


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
    return dState, dDistance

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
        if dDistance[i] < 1.1 and dState[i] == True:
            return True, i
    
    return False, -1    

# Get goal point vector. Returns angle and distance
# Output
# goal_angle: angle to the goal point in radians
#             Positive angle mean goal point is on the right of vehicle, negative mean goal point is on the left
# goal_distance: distance to the goal point
def getGoalPoint():
    _, vehPos = vrep.simxGetObjectPosition( clientID, vehicle_handle, -1, vrep.simx_opmode_blocking)            
    _, dummyPos = vrep.simxGetObjectPosition( clientID, dummy_handle, -1, vrep.simx_opmode_blocking)            

    # Calculate the distance
    delta_distance = np.array(dummyPos) - np.array(vehPos)
    goal_distance = np.linalg.norm(delta_distance)

    # calculate angle
    goal_angle = math.atan( delta_distance[0]/delta_distance[1] )       # delta x / delta y
    

    return goal_angle, goal_distance

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
    MAX_STEER = 45
    MIN_STEER = -45

    # Delta of angle between each action
    action_delta = (MAX_STEER - MIN_STEER) / (options.ACTION_DIM-1)

    # Calculate desired angle
    desired_angle = MAX_STEER - np.argmax(action) * action_delta

    # Set steering position
    setMotorPosition(clientID, steer_handle, desired_angle)
    
    return




###############################33
# TRAINING
#################################33/

# Initialize to original scene
def initScene(vehicle_handle, steer_handle, motor_handle):
    # Reset position of vehicle
    err_code = vrep.simxSetObjectPosition(clientID,vehicle_handle,-1,[0,0,0.2],vrep.simx_opmode_blocking)

    # Reset position of motors & steering
    setMotorPosition(clientID, steer_handle, 0)

    # Reset motor speed
    print("Setting motor speed")
    setMotorSpeed(clientID, motor_handle, 2)
   
    # Read sensor
    dState, dDistance = readSensor(clientID, sensor_handle, vrep.simx_opmode_buffer)         # try it once for initialization

def train():
    # Define placeholders to catch inputs and add options
    options = get_options()
    agent = QAgent(options)
    sess = tf.InteractiveSession()
    
    obs, Q1 = agent.add_value_net(options)
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])
    next_obs, Q2 = agent.add_value_net(options)
    
    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1)
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)
    
    sess.run(tf.initialize_all_variables())
    
    # saving and loading networks
#    saver = tf.train.Saver()
#    checkpoint = tf.train.get_checkpoint_state("checkpoints-cartpole")
#    if checkpoint and checkpoint.model_checkpoint_path:
#        saver.restore(sess, checkpoint.model_checkpoint_path)
#        print("Successfully loaded:", checkpoint.model_checkpoint_path)
#    else:
#        print("Could not find old network weights")
    
    # Some initial local variables
    feed = {}
    eps = options.INIT_EPS
    global_step = 0
    exp_pointer = 0
    learning_finished = False
    
    # The replay memory
    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    
    # Score cache
    score_queue = []

    # The episode loop
    for i_episode in range(options.MAX_EPISODE):
        
        observation = env.reset()
        done = False
        score = 0
        sum_loss_value = 0
        
        # The step loop
        while not done:
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY

#            env.render()
            
            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs : np.reshape(observation, (1, -1))}, eps, options)
            act_queue[exp_pointer] = action
            observation, reward, done, _ = env.step(np.argmax(action))
            
            score += reward
            reward = score  # Reward will be the accumulative score
            
            if done and score<200 :
                reward = -500   # If it fails, punish hard
                observation = np.zeros_like(observation)
            
            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation
    
            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0 # Refill the replay memory if it is full
    
            if global_step >= options.MAX_EXPERIENCE:
                print("Started Training!")
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                feed.update({obs : obs_queue[rand_indexs]})
                feed.update({act : act_queue[rand_indexs]})
                feed.update({rwd : rwd_queue[rand_indexs]})
                feed.update({next_obs : next_obs_queue[rand_indexs]})

                step_loss_value, _ = sess.run([loss, train_step], feed_dict = feed)
                sum_loss_value += step_loss_value
    
        print("====== Episode {} ended with score = {}, avg_loss = {}======".format(i_episode+1, score, sum_loss_value / score))
        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) > 195: # The threshold of being solved
                learning_finished = True
            else:
                learning_finished = False
        if learning_finished:
            print("Testing !!!")
        # save progress every 100 episodes
        if learning_finished and i_episode % 100 == 0:
            saver.save(sess, 'checkpoints-cartpole/' + GAME + '-dqn', global_step = global_step)

########################
# MAIN
########################
if __name__ == "__main__":
    options = get_options()

    # Get client ID
    vrep.simxFinish(-1) 
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

    if clientID == -1:
        print("ERROR: Cannot establish connection to vrep.")
        sys.exit()

    # start simulation
    vrep.simxSynchronous(clientID,True)

    #####################
    # GET HANDLES
    #####################

    print("Getting handles...")
    # Get Motor Handles
    _,h1  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorLeft", vrep.simx_opmode_blocking)
    _,h2  = vrep.simxGetObjectHandle(clientID, "nakedCar_motorRight", vrep.simx_opmode_blocking)
    _,h3  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisRight", vrep.simx_opmode_blocking)
    _,h4  = vrep.simxGetObjectHandle(clientID, "nakedCar_freeAxisLeft", vrep.simx_opmode_blocking)
    _,h5  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringLeft", vrep.simx_opmode_blocking)
    _,h6  = vrep.simxGetObjectHandle(clientID, "nakedCar_steeringRight", vrep.simx_opmode_blocking)

    motor_handle = [h1, h2]
    steer_handle = [h5, h6]

    # Get Sensor Handles
    sensor_handle = []
    for i in range(0,SENSOR_COUNT):
        _,temp_handle  = vrep.simxGetObjectHandle(clientID, "Proximity_sensor" + str(i), vrep.simx_opmode_blocking)
        sensor_handle.append(temp_handle)

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
    options = get_options()
    agent = QAgent(options)
    sess = tf.InteractiveSession()
    
    obs, Q1 = agent.add_value_net(options)
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])
    next_obs, Q2 = agent.add_value_net(options)
    
    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1)
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)
    
    sess.run(tf.initialize_all_variables())

    # saving and loading networks
    if options.USE_SAVE = true:

        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("checkpoints-vehicle")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


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
        

    # EPISODE LOOP
    for j in range(0,options.MAX_EPISODE): 
        print("Episode: " + str(j))

        sum_loss_value = 0
        vehPosDataTrial = np.array([0,0,0.2])        # Initialize data
        done = 0

        # Start simulation and initilize scene
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
        initScene(vehicle_handle, steer_handle, motor_handle)

        for i in range(0,options.MAX_TIMESTEP):     # Time Step Loop
            if i % 10 == 0:
                print("\tStep:" + str(i))

            # Decay epsilon
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY

            # get data
            dDistance, dState, gInfo, vehPos = getVehicleState()
            observation = dState + dDistance + gInfo
            #print(np.round(observation,2))

            # Save data
#            sensorData = np.vstack( (sensorData,dDistance) )
#            sensorDetection = np.vstack( (sensorDetection,dState) )
#            vehPosDataTrial = np.vstack( (vehPosDataTrial, vehPos) )

            # Update memory
            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs : np.reshape(observation, (1, -1))}, eps, options)
            act_queue[exp_pointer] = action

            # Apply action
            applySteeringAction( action )
#            observation, reward, done, _ = env.step(np.argmax(action))
            if options.manual == True:
                input('Press any key to step forward.')

            # Step simulation by one step
            vrep.simxSynchronousTrigger(clientID);

            # Get new Data
            dDistance, dState, gInfo, vehPos = getVehicleState()
            observation = dState + dDistance + gInfo
            reward = -0.01*gInfo[1]**2         # cost is the distance squared

            # If vehicle collided, give large negative reward
            if detectCollision(dDistance,dState)[0] == True:
                print('Vehicle collided!')
                print(dDistance)
                print(dState)
                vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)

                # Wait until simulation is stopped.
                simulation_status = 1
                bit_mask = 1
                while bit_mask & simulation_status != 0:       # Get right-most bit and check if it is 1
                    _, simulation_status = vrep.simxGetInMessageInfo(clientID, vrep.simx_headeroffset_server_state)
#                    print(bit_mask & simulation_status)
#                    print("{0:b}".format(simulation_status))
                    time.sleep(0.1)
    
                # Save trial data
#                vehPosData.append( vehPosDataTrial )

                # Set flag and reward
                done = 1
                reward = -1e6

            rwd_queue[exp_pointer] = reward
            next_obs_queue[exp_pointer] = observation
    
            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0 # Refill the replay memory if it is full
  
            if global_step >= options.MAX_EXPERIENCE:
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                feed.update({obs : obs_queue[rand_indexs]})
                feed.update({act : act_queue[rand_indexs]})
                feed.update({rwd : rwd_queue[rand_indexs]})
                feed.update({next_obs : next_obs_queue[rand_indexs]})
                
                step_loss_value, _ = sess.run([loss, train_step], feed_dict = feed)
                # Use sum to calculate average loss of this episode
                sum_loss_value += step_loss_value


            # If collided, end this episode
            if done == 1:
                break

        # EPISODE ENDED
        print("====== Episode {} ended.".format(j))
        
        # save progress every 10 episodes
        if j % 10 == 0:
            saver.save(sess, 'checkpoints-vehicle/vehicle-dqn', global_step = global_step)

    # stop the simulation & close connection
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    vrep.simxFinish(clientID)





    #############################3
    # Visualize Data
    ##############################3

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


    # Plot position of vehicle
    plt.figure(1)

#    for i in range(0,4):
#        plt.scatter(vehPosData[i][:,0],vehPosData[i][:,1], label="Trial #" + str(i))

    # Plot Map
    plt.plot([1.25, 1.25],[0, 60],'k')
    plt.plot([-3.75, -3.75],[0, 60],'k')
    plt.plot([-1.25, -1.25],[0, 60],'k')
    plt.plot([0, 0],[0, 60],'k--')
    plt.plot([-2.5, -2.5],[0, 60],'k--')
    plt.plot([0],[60],'xr') 

    # Plot properties
    plt.axis( (-3.75, 1.25, 0, 80)   )
    plt.legend()
    plt.title("Vehicle Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()













