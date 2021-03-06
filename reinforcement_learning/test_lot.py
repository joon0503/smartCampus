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
from argparse import ArgumentParser

# From other files
from utils.utils_vrep import *                  # import everything directly from utils_vrep.py
from utils.utils_training import *              # import everything directly from utils_vrep.py
from utils.scene_constants import scene_constants 
from utils.rl_dqn import QAgent
from utils.rl_icm import ICM
from utils.experience_replay import SumTree 
from utils.experience_replay import Memory
from utils.utils_data import data_pack
from utils.scene_ParkingLot import *

def get_options():
    parser = ArgumentParser(
        description='File for learning'
        )
    parser.add_argument('--MAX_EPISODE', type=int, default=10001,
                        help='max number of episodes iteration\n')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=1000,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=11,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.98,
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
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=1000,
                        help='size of experience replay memory')
    parser.add_argument('--SAVER_RATE', type=int, default=20000,
                        help='Save network after this number of episodes')
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
    parser.add_argument('--enable_PER', action='store_true', default = False,
                        help='Enable the usage of PER.')
    parser.add_argument('--disable_duel', action='store_true',
                        help='Disable the usage of double network.')
    parser.add_argument('--FRAME_COUNT', type=int, default=1,
                        help='Number of frames to be used')
    parser.add_argument('--ACT_FUNC', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--GOAL_REW', type=int, default=500,
                        help='Activation function')
    parser.add_argument('--FAIL_REW', type=int, default=-2000,
                        help='Activation function')
    parser.add_argument('--VEH_COUNT', type=int, default=10,
                        help='Number of vehicles to use for simulation')
    parser.add_argument('--INIT_SPD', type=int, default=10,
                        help='Initial speed of vehicle in  km/hr')
    parser.add_argument('--DIST_MUL', type=int, default=20,
                        help='Multiplier for rewards based on the distance to the goal')
    parser.add_argument('--EXPORT', action='store_true', default=False,
                        help='Export the weights into a csv file')
    parser.add_argument('--SEED', type=int, default=1,
                        help='Set simulation seed')
    parser.add_argument('--CTR_FREQ', type=float, default=0.2,
                        help='Control frequency in seconds. Upto 0.001 seconds')
    parser.add_argument('--SIM_STEP', type=float, default=0.025,
                        help='VREP simulation time step in seconds. Upto 0.001 seconds')
    parser.add_argument('--MIN_LIDAR_CONST', type=float, default=0.2,
                        help='Stage-wise reward 1/(min(lidar)+MIN_LIDAR_CONST) related to minimum value of LIDAR sensor')
    parser.add_argument('--POINT_CNT', type=int, default=5,
                        help='# of way point')
    options = parser.parse_args()

    return parser, options


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
        sensor_stack    = np.concatenate(sensor_queue)[0:scene_const.sensor_count*options.FRAME_COUNT]
        goal_stack      = np.concatenate(goal_queue)[0:2*options.FRAME_COUNT]
        observation     = np.concatenate((sensor_stack, goal_stack))    
    else:
        sensor_stack    = np.concatenate(sensor_queue)[scene_const.sensor_count*options.FRAME_COUNT:]
        goal_stack      = np.concatenate(goal_queue)[2*options.FRAME_COUNT:]
        observation     = np.concatenate((sensor_stack, goal_stack))    

    return observation

# Calculate Approximate Rewards for variaous cases
def printRewards( scene_const, options ):
    # Some parameters
    veh_speed = options.INIT_SPD/3.6  # 10km/hr in m/s

    # Time Steps
    #dt_code = scene_const.dt * options.FIX_INPUT_STEP

    # Expected Total Time Steps
    total_step = (scene_const.goal_distance/veh_speed)*(1/options.CTR_FREQ)

    # Reward at the end
    rew_end = 0
    for i in range(0, int(total_step)):
        goal_distance = scene_const.goal_distance - i*options.CTR_FREQ*veh_speed 
        if i != total_step-1:
            rew_end = rew_end -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_end = rew_end + options.GOAL_REW*(options.GAMMA**i) 

    # Reward at Obs
    rew_obs = 0
    for i in range(0, int(total_step*0.5)):
        goal_distance = scene_const.goal_distance - i*options.CTR_FREQ*veh_speed 
        if i != int(total_step*0.5)-1:
            rew_obs = rew_obs -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_obs = rew_obs + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 75%
    rew_75 = 0
    for i in range(0, int(total_step*0.75)):
        goal_distance = scene_const.goal_distance - i*options.CTR_FREQ*veh_speed 
        if i != int(total_step*0.75)-1:
            rew_75 = rew_75 -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_75 = rew_75 + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 25%
    rew_25 = 0
    for i in range(0, int(total_step*0.25)):
        goal_distance = scene_const.goal_distance - i*options.CTR_FREQ*veh_speed 
        if i != int(total_step*0.25)-1:
            rew_25 = rew_25 -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
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
    print("Control Frequency (s)  : ", options.CTR_FREQ)
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

def printSpdInfo():
    desiredSpd = options.INIT_SPD

    wheel_radius = 0.63407*0.5      # Wheel radius in metre

    desiredSpd_rps = desiredSpd*(1000/3600)*(1/wheel_radius)   # km/hr into radians per second

    print("Desired Speed: " + str(desiredSpd) + " km/hr = " + str(desiredSpd_rps) + " radians per seconds = " + str(math.degrees(desiredSpd_rps)) + " degrees per seconds. = " + str(desiredSpd*(1000/3600)) + "m/s" )

    return

###############################33
# TRAINING
#################################33/




########################
# MAIN
########################
if __name__ == "__main__":
    # Print Options
    parser, options = get_options()
    print(str(options).replace(" ",'\n'))

    # Define fix input step
    FIX_INPUT_STEP = int( (1000*options.CTR_FREQ)/(1000*options.SIM_STEP) )
    print('FIX_INPUT_STEP : ', FIX_INPUT_STEP)
     
    if int(1000*options.CTR_FREQ) % int(1000*options.SIM_STEP) != 0:
        raise ValueError('CTR_FREQ must be a integer multiple of SIM_STEP! Currently CTR_FREQ % SIM_STEP = ' + str(options.CTR_FREQ % options.SIM_STEP) )

    # Set Seed
    np.random.seed(1)
    random.seed(1)
    tf.set_random_seed(1)

    # Set print options
    np.set_printoptions( precision = 4, linewidth = 100 )

    # For interactive plot
    plt.ion()

    ######################################33
    # SET 'GLOBAL' Variables
    ######################################33
    START_TIME       = datetime.datetime.now() 
    START_TIME_STR   = str(START_TIME).replace(" ","_")
    START_TIME_STR   = str(START_TIME).replace(":","_")

    # Randomize obstacle position at initScene
    RANDOMIZE = True

    # Get client ID
    vrep.simxFinish(-1) 
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
    if clientID == -1:
        print("ERROR: Cannot establish connection to vrep.")
        sys.exit()

    # Set scene_constants
    scene_const = scene_constants()
    #scene_const.max_distance        = 20
    #scene_const.sensor_count        = 9
    scene_const.clientID            = clientID
    #scene_const.max_steer           = 15
    #scene_const.collision_distance  = 1.3
    #scene_const.goal_distance       = 60
    scene_const.dt                  = options.SIM_STEP         #simulation time step

    # Save options
    if not os.path.exists("./checkpoints-vehicle"):
        os.makedirs("./checkpoints-vehicle")
    option_file = open("./checkpoints-vehicle/options_"+START_TIME_STR+'.txt', "w")

    # For each option
    for x in sorted(vars(options).keys()):
        option_file.write( str(x).ljust(20) + ": " + str(vars(options)[x]).ljust(10) )   # write option
        if vars(options)[x] == parser.get_default(x):       # if default value
            option_file.write( '(DEFAULT)' )                # say it is default
        option_file.write('\n')
    option_file.close()

    # Print Rewards
    printRewards( scene_const, options )

    # Print Speed Infos
    printSpdInfo()



    # Set Sampling time
    vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, scene_const.dt , vrep.simx_opmode_oneshot)

    # start simulation
    vrep.simxSynchronous(clientID,True)

    #####################
    # GET HANDLES
    #####################

    print("Getting handles...")
    motor_handle, steer_handle = getMotorHandles( options, scene_const )
    sensor_handle = getSensorHandles( options, scene_const )

    # Get vehicle handle
    vehicle_handle = np.zeros(options.VEH_COUNT, dtype=int)
    for i in range(0,options.VEH_COUNT):
        err_code, vehicle_handle[i] = vrep.simxGetObjectHandle(clientID, "dyros_vehicle" + str(i), vrep.simx_opmode_blocking)

    # Get goal point handle
    dummy_handle = np.zeros(options.POINT_CNT, dtype=int)
    for i in range(0,options.POINT_CNT):
        err_code, dummy_handle[i] = vrep.simxGetObjectHandle(clientID, "GoalPoint" + str(i), vrep.simx_opmode_blocking)

    # Make Large handle list for communication
    handle_list = [options.VEH_COUNT, scene_const.sensor_count] 

    # Make handles into big list
    for v in range(0,options.VEH_COUNT):
        handle_list = handle_list + [vehicle_handle[v]] + sensor_handle[v].tolist() + [dummy_handle[v]]

    # Make handle into dict to be passed around functions
    handle_dict = {
        'sensor'    : sensor_handle,
        'motor'     : motor_handle,
        'steer'     : steer_handle,
        'vehicle'   : vehicle_handle,
        'dummy'     : dummy_handle,
    }

    ########################
    # Initialize Test Scene
    ########################

    # Data
    sensorData = np.zeros(scene_const.sensor_count)   
    sensorDetection = np.zeros(scene_const.sensor_count)   
    #vehPosData = []

    ##############
    # TF Setup
    ##############
    # Define placeholders to catch inputs and add options
    agent_train     = QAgent(options,scene_const, 'Training')
    agent_target    = QAgent(options,scene_const, 'Target')
    agent_icm       = ICM(options,scene_const,'icm_Training')

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
    feed_icm = {}
    eps = options.INIT_EPS
    global_step = 0

    # The replay memory.
    if options.enable_PER == False:
        # Don't use PER
        print("=================================================")
        print("NOT using PER!")
        print("=================================================")
        replay_memory = Memory(options.MAX_EXPERIENCE)
    else:
        # Use PER
        print("=================================================")
        print("Using PER!")
        print("=================================================")
        replay_memory = Memory(options.MAX_EXPERIENCE, disable_PER = False, absolute_error_upperbound = 2000)
        

    printTFvars()

    # Export weights
    if options.EXPORT == True:
        print("Exporting weights...")
        val = tf.trainable_variables(scope='Training')
        for v in val:
            fixed_name = str(v.name).replace('/','_')
            fixed_name = str(v.name).replace(':','_')
            np.savetxt( './model_weights/' + fixed_name + '.txt', sess.run([v])[0], delimiter=',')
        sys.exit() 


    ########################
    # END TF SETUP
    ########################

    ###########################        
    # DATA VARIABLES
    ###########################        
    data_package = data_pack( START_TIME_STR )    

    reward_data         = np.empty(0)
    avg_loss_value_data = np.empty(1)
#    veh_pos_data        = np.empty([options.MAX_EPISODE, options.MAX_TIMESTEP, 2])
    avg_epi_reward_data = np.zeros(options.MAX_EPISODE)
    #epi_reward_data = np.zeros(0)
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

    initScene( scene_const, options, list(range(0,options.VEH_COUNT)), handle_dict, randomize = RANDOMIZE)               # initialize

    # List of deque to store data
    sensor_queue = []
    goal_queue   = []
    for i in range(0,options.VEH_COUNT):
        sensor_queue.append( deque() )
        goal_queue.append( deque() )

    # initilize them with initial data
    veh_pos, veh_heading, dDistance, gInfo = getVehicleStateLUA( handle_list, scene_const )
    sensor_queue, goal_queue = initQueue( options, sensor_queue, goal_queue, dDistance, gInfo )

    # Current Waypoint Target
    curr_pnt = 0

    # Global Step Loop
    while epi_counter <= options.MAX_EPISODE:
        #GS_START_TIME_STR   = datetime.datetime.now()
        # Decay epsilon
        global_step += options.VEH_COUNT
        if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
            eps = eps * options.EPS_DECAY

        print('Current Target : ' + str(curr_pnt) )


        ####
        # Find & Apply Action
        ####
        # FIXME: This can be made faster by not using loop and just pushing through network all at once
        for v in range(0,options.VEH_COUNT):
            # Get current info to generate input
            observation     = getObs( sensor_queue[v], goal_queue[v], old=False)

            # Get Action. First vehicle always exploit
            if v == 0:
                #print('curr_state:', observation)
                action_stack[v] = agent_train.sample_action(
                                                    {
                                                        agent_train.observation : np.reshape(observation, (1, -1))
                                                    },
                                                    eps,
                                                    options,
                                                    False
                                                 )
            else:
                action_stack[v] = agent_train.sample_action(
                                                    {
                                                        agent_train.observation : np.reshape(observation, (1, -1))
                                                    },
                                                    eps,
                                                    options,
                                                    False
                                                 )

        applySteeringAction( action_stack, options, handle_dict, scene_const )

        ####
        # Step
        ####

        for q in range(0,FIX_INPUT_STEP):
            vrep.simxSynchronousTrigger(clientID);

        if options.manual == True:
            input('Press Enter')

        ####
        # Get Next State
        ####
        next_veh_pos, next_veh_heading, next_dDistance, _ = getVehicleStateLUA( handle_list, scene_const )

        retCode, goal_pos = vrep.simxGetObjectPosition(clientID, dummy_handle[curr_pnt], -1, vrep.simx_opmode_blocking)

        next_gInfo = np.zeros((1,2))
        next_gInfo[0,:] = getGoalInfo( next_veh_pos, goal_pos[0:2], scene_const )

        for v in range(0,options.VEH_COUNT):
            # Get new Data
            #veh_pos_queue[v].append(next_veh_pos[v])   

            # Update queue
            sensor_queue[v].append(next_dDistance[v])
            sensor_queue[v].popleft()
            goal_queue[v].append(next_gInfo[v])
            goal_queue[v].popleft()

            # Get reward for each vehicle
            reward_stack[v] = -(options.DIST_MUL+1/(next_dDistance[v].min()+options.MIN_LIDAR_CONST))*next_gInfo[v][1]**2
            # cost is the distance squared + inverse of minimum LIDAR distance
        #######
        # Test Estimation
        #######
        if options.TESTING == True:
            v = 0

            # Print curr & next state
            curr_state     = getObs( sensor_queue[v], goal_queue[v], old=True)
            next_state     = getObs( sensor_queue[v], goal_queue[v], old=False)
            #print('curr_state:', curr_state)
            #print('next_state:', next_state)
            #print('estimate  :', agent_icm.getEstimate( {agent_icm.observation : np.reshape(np.concatenate([curr_state, action_stack[v]]), [-1, 16])  }  ) )
            #print('')
            agent_icm.plotEstimate( curr_state, action_stack[v], next_veh_heading[v], scene_const, agent_train, options, save=True)
            
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
            collision_detected, collision_sensor = detectCollision(next_dDistance[v], scene_const)
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
                curr_pnt = 0

                # If collided, skip checking for goal point
                continue

            # If vehicle is at the goal point, give large positive reward
            if detectReachedGoal(next_veh_pos[v], next_gInfo[v], next_veh_heading[v], scene_const):
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' reached goal point' + str(curr_pnt))
                print('-----------------------------------------')

                # Increment goal point. If over 5, start over               
                curr_pnt = ( curr_pnt + 1 ) % options.POINT_CNT

                # Reset Simulation
                #reset_veh_list.append(v)

                # Set flag and reward
                #reward_stack[v] = options.GOAL_REW
                #eps_tracker[epi_counter] = eps
                #epi_counter += 1

                # Set done
                #epi_done[v] = 1

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
        
        ###########
        # START LEARNING
        ###########

        # Add latest information to memory
        for v in range(0,options.VEH_COUNT):
            # Get observation
            observation             = getObs( sensor_queue[v], goal_queue[v], old = True)
            next_observation        = getObs( sensor_queue[v], goal_queue[v], old = False)

            # Add experience. (observation, action in one hot encoding, reward, next observation, done(1/0) )
            #experience = observation, action_stack[v], reward_stack[v], next_observation, epi_done[v]
            experience = observation, np.argmax(action_stack[v]), reward_stack[v], next_observation, epi_done[v]
            #print('experience', experience)
           
            # Save new memory 
            replay_memory.store(experience)

        # Reset Vehicles    
        initScene( scene_const, options, reset_veh_list, handle_dict, randomize = RANDOMIZE)               # initialize

        # Reset data queue
        _, _, reset_dDistance, reset_gInfo = getVehicleStateLUA( handle_list, scene_const )
        sensor_queue, goal_queue = resetQueue( options, sensor_queue, goal_queue, reset_dDistance, reset_gInfo, reset_veh_list )

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
            print('\tEpisode #: ' + str(epi_counter) + ' / ' + str(options.MAX_EPISODE) + '\n\tStep: ' + str(int(epi_step_stack[v])) )
            print('\tEpisode Reward: ' + str(epi_reward_stack[v])) 
            print('Last Loss: ',data_package.avg_loss[-1])
            print('========')
            print('')

        # Update Counters
        epi_step_stack = epi_step_stack + 1

        # Reset rewards for finished vehicles
        for v in reset_veh_list:
            data_package.add_reward( epi_reward_stack[v] )  # Add reward
            data_package.add_eps( eps )                     # Add epsilon used for this reward

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
            initScene( scene_const, options, list(range(0,options.VEH_COUNT)), handle_dict, randomize = RANDOMIZE)               # initialize

            # reset data queue
            _, _, reset_dDistance, reset_gInfo = getVehicleStateLUA( handle_list, scene_const )
            sensor_queue, goal_queue = resetQueue( options, sensor_queue, goal_queue, reset_dDistance, reset_gInfo, list(range(0,options.VEH_COUNT)) )
        
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
                data_package.save_reward()
                data_package.save_loss()

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

    # Plot Reward
    data_package.plot_reward( options.RUNNING_AVG_STEP )
    data_package.plot_loss()


# END
