# DQN model to solve vehicle control problem
import datetime
import math
import os
import pickle
import random
import re
import sys
import time
from argparse import ArgumentParser
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import tensorflow as tf
from icecream import ic

from utils.env_py import *
from utils.experience_replay import Memory, SumTree
from utils.q_algorithm import dqn
from utils.rl_dqn import QAgent
from utils.rl_icm import ICM
from utils.scene_constants_pb import scene_constants
from utils.utils_data import data_pack
from utils.utils_pb import (controlCamera, drawDebugLines)


def get_options():
    # Parser Settings
    parser = ArgumentParser(
        description='File for learning'
        )
    parser.add_argument('--MAX_EPISODE', type=int, default=5000,
                        help='max number of episodes iteration\n')
    parser.add_argument('--MAX_TIMESTEP', type=int, default=250,
                        help='max number of time step of simulation per episode')
    parser.add_argument('--ACTION_DIM', type=int, default=5,
                        help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=11,
                        help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.995,
                        help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=1.0,
                        help='initial probability for randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-3,
                        help='finial probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.995,
                        help='epsilon decay rate')
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=10800,
                        help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=20000,
                        help='size of experience replay memory')
    parser.add_argument('--SAVER_RATE', type=int, default=500,
                        help='Save network after this number of episodes')
    parser.add_argument('--TARGET_UPDATE_STEP', type=int, default=3000,
                        help='Number of steps required for target update')
    parser.add_argument('--BATCH_SIZE', type=int, default=32,
                        help='mini batch size'),
    parser.add_argument('--H1_SIZE', type=int, default=160,
                        help='size of hidden layer 1')
    parser.add_argument('--H2_SIZE', type=int, default=160,
                        help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=160,
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
    parser.add_argument('--enable_PER', action='store_true', default = False,
                        help='Enable the usage of PER.')
    parser.add_argument('--enable_ICM', action='store_true', default = False,
                        help='Enable the prediction network.')
    parser.add_argument('--enable_GUI', action='store_true', default = False,
                        help='Enable the GUI.'),
    parser.add_argument('--enable_TRAJ', action='store_true', default = False,
                        help='Generate and print estimated trajectory.'),
    parser.add_argument('--ADD_NOISE', action='store_true', default = False,
                        help='Add noise to the sensor measurement.'),
    parser.add_argument('--VERBOSE', action='store_true', default = False,
                        help='Verbose output')
    parser.add_argument('--disable_duel', action='store_true',
                        help='Disable the usage of double network.')
    parser.add_argument('--FRAME_COUNT', type=int, default=4,
                        help='Number of frames to be used')
    parser.add_argument('--ACT_FUNC', type=str, default='relu',
                        help='Activation function')
    parser.add_argument('--GOAL_REW', type=int, default=4000,
                        help='Activation function')
    parser.add_argument('--FAIL_REW', type=int, default=-5000,
                        help='Activation function')
    parser.add_argument('--VEH_COUNT', type=int, default=6,
                        help='Number of vehicles to use for simulation')
    parser.add_argument('--INIT_SPD', type=int, default=50,
                        help='Initial speed of vehicle. 100 -> 10m/s = 36km/hr')
    parser.add_argument('--DIST_MUL', type=int, default=10,
                        help='Multiplier for rewards based on the distance to the goal')
    parser.add_argument('--EXPORT', action='store_true', default=False,
                        help='Export the weights into a csv file')
    parser.add_argument('--SEED', type=int, default=1,
                        help='Set simulation seed')
    parser.add_argument('--CTR_FREQ', type=float, default=0.2,
                        help='Control frequency in seconds. Upto 0.001 seconds')
    parser.add_argument('--MIN_LIDAR_CONST', type=float, default=0.075,
                        help='Stage-wise reward 1/(min(lidar)+MIN_LIDAR_CONST) related to minimum value of LIDAR sensor')
    parser.add_argument('--L2_LOSS', type=float, default=0.0,
                        help='Scale of L2 loss')
    parser.add_argument('--NOISE_PROB', type=float, default=0.2,
                        help='Probability of sensor value being modified (if enabled)')
    parser.add_argument('--FIX_INPUT_STEP', type=int, default=6,
                        help='Fix input steps')
    parser.add_argument('--X_COUNT', type=int, default=6,
                        help='Width of the grid for vehicles. Default of 5 means we lay down 5 vehicles as the width of the grid')
    parser.add_argument('--THREAD', type=int, default=1,
                        help='Number of threads for parallel simulation.')
    parser.add_argument('--WEIGHT_FILE', type=str, default=None,
                        help='Relative path to the weight file to load. Only works for KERAS.')
    parser.add_argument('--DUMP_OPTIONS', action='store_true', default = False,
                        help='Dump options and scene_const files.')
    parser.add_argument('--DRAW', action='store_true', default = False,
                        help='Visualize the first vehicle')
    options = parser.parse_args()

    # Check Inputs
    if options.TARGET_UPDATE_STEP % options.VEH_COUNT != 0:
        raise ValueError('VEH_COUNT must divide TARGET_UPDATE_STEPS')

    if options.EPS_ANNEAL_STEPS % options.VEH_COUNT != 0:
        raise ValueError('VEH_COUNT must divide EPS_ANNEAL_STEPS')

    # Save options
    if not os.path.exists("./checkpoints-vehicle"):
        os.makedirs("./checkpoints-vehicle")

    if options.TESTING == True:
        option_file = open("./checkpoints-vehicle/options_TESTING_"+START_TIME_STR+'.txt', "w")
    else:
        option_file = open("./checkpoints-vehicle/options_"+START_TIME_STR+'.txt', "w")

    # For each option
    for x in sorted(vars(options).keys()):
        # Option string
        opt_str = str(x).ljust(20) + ": " + str(vars(options)[x]).ljust(10)
        if vars(options)[x] == parser.get_default(x):
            opt_str = opt_str + '(DEFAULT)'

        # Print to file
        option_file.write( opt_str )   # write option
        option_file.write('\n')

        # Print to terminal
        print( opt_str )
    option_file.close()

    return parser, options

# Print version for main tools used in the script
def printVersions():
    print('===============================================')
    print('TENSOR FLOW VERSION : ' + tf.VERSION)
    print('KERAS VERSION       : ' + tf.keras.__version__)
    print('PYTHON VERSION      : ' + sys.version)
    print('PyBullet Version:   : ' + str(p.getAPIVersion()) )
    print('Numpy Version       : ' + str(np.version.version))
    print('===============================================')

    return

# Dump options & scene_const
def dumpOptions( options, scene_const ):
    fileName = 'genTraj_options_file'

    out_dict = {'options' : options, 'scene_const' : scene_const}
    outFile = open( fileName, 'wb')
    pickle.dump(out_dict, outFile)
    outFile.close()

    infile = open(fileName,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    print(new_dict)
    return

########################
# MAIN
########################
if __name__ == "__main__":
    # Debugging Options
    ic.configureOutput(includeContext=True)

    ##############################
    # SET 'GLOBAL' Variables
    ##############################

    # Start time
    START_TIME       = datetime.datetime.now()
    START_TIME_STR   = re.sub( r'\ |:', '_', str( START_TIME) )

    # Randomize obstacle position at initScene
    RANDOMIZE = True

    ##############################
    # Input parsing
    ##############################

    # print versions
    printVersions()

    # Parse options
    _, options = get_options()

    # Set Seed
    np.random.seed( options.SEED )
    random.seed( options.SEED )
    tf.set_random_seed( options.SEED )

    # Set print options
    np.set_printoptions( precision = 4, linewidth = 100 )


    ######################################
    # Simulation Start
    ######################################

    # Initial Camera position
    cam_pos = [0,0,0]
    cam_dist = 10

    # Start Environment
    sim_env = env_py( options, scene_constants() )
    sim_env.scene_const.clientID, handle_dict = sim_env.start()

    # Check Dump
    if options.DUMP_OPTIONS == True:
        dumpOptions(options, sim_env.scene_const)
        sys.exit()

    # Print Infos
    sim_env.printInfo()

    # Draw initial lines
    if options.enable_GUI == True:
        sensor_handle       = drawDebugLines( options, sim_env.scene_const, handle_dict, createInit = True )
        collision_handle    = drawDebugLines( options, sim_env.scene_const, handle_dict, createInit = True )

    # Print Handles
    if options.VERBOSE == True:
        ic('handle_dict',handle_dict)

    ##############
    # TF Setup
    ##############
    # FIXME: Currently using init_eps as the initial value of the curriculum hardness. Separate this into a new option
    q_algo          = dqn( sim_env, options.INIT_EPS, load = True )
    if options.enable_ICM == True:
        agent_icm       = ICM(options,sim_env.scene_const,'icm_Training')

    # Some initial local variables
    feed_icm        = {}
    global_step     = 0

    ###########################        
    # DATA VARIABLES
    ###########################        
    data_package = data_pack( START_TIME_STR )    

    ###########################        
    # Initialize Variables
    ###########################        
    epi_counter         = 0                                                             # Counts # of finished episodes
    last_saved_epi      = 0                                                             # variable used for checking when to save
    case_direction      = np.zeros(options.VEH_COUNT)                                   # store tested direction

    # Initialize Scene
    _, _, _ = sim_env.initScene( list(range(0,options.VEH_COUNT)), RANDOMIZE )

    # initilize them with initial data
    sim_env.updateObservation( range(0,sim_env.options.VEH_COUNT) )

    # Global Step Loop
    while epi_counter <= options.MAX_EPISODE:
        # GS_START_TIME_STR   = datetime.datetime.now()
        if options.enable_GUI == True and options.manual == False:
            cam_pos, cam_dist = controlCamera( cam_pos, cam_dist )  
            time.sleep(0.01)

        global_step += options.VEH_COUNT

        ####
        # Find & Apply Action
        ####
        obs_sensor_stack = np.empty((options.VEH_COUNT, sim_env.scene_const.sensor_count*2, options.FRAME_COUNT))
        obs_goal_stack   = np.empty((options.VEH_COUNT, 2, options.FRAME_COUNT))


        # Get observation stack (which is used in getting the action) 
        _, _, obs_sensor_stack, obs_goal_stack = sim_env.getObservation(old = False)
        # obs_sensor_stack = addNoise(options, sim_env.scene_const, obs_sensor_stack)

        if options.TESTING == True and options.VERBOSE == True:
            ic(sim_env.sensor_queue, obs_sensor_stack)
            ic(sim_env.goal_queue, obs_goal_stack)

        if options.DRAW == True:
            sim_env.plotVehicle(save=True)

        # Get optimal action q_algo
        action_feed = {}
        action_feed.clear()
        action_feed.update({'observation_sensor_k': obs_sensor_stack[:,0:sim_env.scene_const.sensor_count,:]})
        action_feed.update({'observation_state': obs_sensor_stack[:,sim_env.scene_const.sensor_count:,:]})
        action_feed.update({'observation_goal_k': obs_goal_stack})
        targetSteer_k, action_stack_k = q_algo.getOptimalAction( action_feed )

        # Apply Action
        sim_env.applyAction( targetSteer_k )

        ####
        # Step
        ####
        sim_env.step()

        # Update Observation
        sim_env.updateObservation( range(0,sim_env.options.VEH_COUNT), add_noise = sim_env.options.ADD_NOISE )


        ####
        # Get Next State
        ####
        next_veh_pos, next_veh_heading, next_dDistance, next_gInfo = sim_env.getObservation( verbosity = options.VERBOSE, frame = -1 )
        # next_dDistance = addNoise(options, sim_env.scene_const, next_dDistance)

        ####
        # Handle Events & Get Rewards
        ####
        reward_stack, veh_status, epi_done, epi_sucess = sim_env.getRewards( next_dDistance, next_veh_pos, next_gInfo, next_veh_heading)

        if sim_env.options.VERBOSE == True and sim_env.options.TESTING == True:
            ic('REWARDS:', reward_stack)


        # List of resetting vehicles
        reset_veh_list = [ v for v in range(0,options.VEH_COUNT) if veh_status[v] != sim_env.scene_const.EVENT_FINE ]

        # Update epi counter
        epi_counter += len(reset_veh_list)

        if options.VERBOSE == True:
            ic(reward_stack)

        # Draw Debug Lines
        if options.enable_GUI == True:
            # Draw Lidar
            sensor_handle = drawDebugLines(options, sim_env.scene_const, handle_dict, next_dDistance, sensor_handle, createInit = False )

            # Draw Collision Range
            collision_handle = drawDebugLines(options, sim_env.scene_const, handle_dict, np.ones((options.VEH_COUNT,sim_env.scene_const.sensor_count))*(sim_env.scene_const.collision_distance/sim_env.scene_const.sensor_distance), collision_handle, createInit = False, ray_width = 4, rayHitColor = [0,0,0] )


        #######
        # Test Estimation
        #######
        if options.TESTING == True and options.enable_ICM == True:
            v = 0

            # Get curr & next state
            _, _, curr_state_sensor, curr_state_goal     = sim_env.getObservation( old = False)

            # Get curr & next state
            # _, _, curr_state_sensor, curr_state_goal     = sim_env.getObservation( old = False)
            # _, _, next_state_sensor, next_state_goal     = sim_env.getObservation( old = True )

            # Print curr & next state
            # ic(curr_state_sensor[v], curr_state_goal[v])
            # ic(next_state_sensor[v], next_state_goal[v])

            # ic(np.expand_dims(curr_state_sensor, axis = 0).shape)
            # Print estimate
            icm_est = agent_icm.getEstimate( 
                        {
                            'icm_input_sensor_frame': np.expand_dims(curr_state_sensor[v], axis = 0),
                            'icm_input_goal_frame'  : np.expand_dims(curr_state_goal[v], axis = 0),
                            'icm_input_action'      : action_stack_k[0].reshape((1,1))
                        }  
                    )
            if options.VERBOSE == True:
                ic(icm_est)

            # agent_icm.plotEstimate( sim_env.scene_const, options, curr_state_sensor, curr_state_goal, action_stack[v], next_veh_heading[v], agent_train, save=True, ref = 'vehicle')

            # Generate trajectory

        # Generate Trajectory
        if options.enable_TRAJ == True:
            max_horizon=5
            next_veh_pos, next_veh_heading, next_state_sensor, next_state_goal     = sim_env.getObservation( old = True )
            gen_traj = q_algo.genTrajectory(next_veh_pos, next_veh_heading, next_state_sensor, next_state_goal, max_horizon)
            ic('GENERATED TRAJECTORY', gen_traj)

        ###########
        # START LEARNING
        ###########

        # Add latest information to memory
        # Get observation
        _, _, observation_sensor, observation_goal            = sim_env.getObservation( verbosity = options.VERBOSE, old = True)
        _, _, next_observation_sensor, next_observation_goal  = sim_env.getObservation( verbosity = options.VERBOSE, old = False)

        for v in range(0,options.VEH_COUNT):
            # Add experience. (observation, action in one hot encoding, reward, next observation, done(1/0) )
            experience = observation_sensor[v], observation_goal[v], action_stack_k[v], reward_stack[v], next_observation_sensor[v], next_observation_goal[v], epi_done[v]
           
            # Save new memory 
            q_algo.replay_memory.store(experience)

        # Start training
        if global_step >= options.MAX_EXPERIENCE and options.TESTING == False:
            for tf_train_counter in range(0,options.VEH_COUNT):
                ##############################
                # Train Control Algorithm
                ##############################
                loss_k, states_sensor_mb, next_states_sensor_mb, states_goal_mb, next_states_goal_mb, actions_mb = q_algo.trainOneStep()

                ##############################
                # Train forward model
                ##############################
                if options.enable_ICM == True:
                    feed_icm.clear()

                    # Feed
                    feed_icm.update(
                        { 
                            'icm_input_sensor_frame' : states_sensor_mb,
                            'icm_input_goal_frame'   : states_goal_mb,
                            'icm_input_action'       : actions_mb*(1/(options.ACTION_DIM-1)),         # action_stack is 0 to ACTION_DIM, scale to 0-1
                        }
                    )

                    # next_statse have BATCH_SIZE x SENSOR_COUNT *2x FRAME_COUNT. Hence, [:,:,-1] pick up the latest frame and 0:sensor_count to only pick up hit fraction and not the detection state
                    icm_target = np.hstack([next_states_sensor_mb[:,0:sim_env.scene_const.sensor_count,-1], next_states_goal_mb[:,:,-1]])    # BATCH_SIZE x (SENSOR_COUNT + 2)

                    # Train
                    loss_icm_k = agent_icm.model.train_on_batch( feed_icm, icm_target )

                # Save LOSS
                data_package.add_loss( loss_k )
        elif global_step < options.MAX_EXPERIENCE:
            # If just running to get memory, do not increment counter
            epi_counter = 0

        handle_dict, sim_env.scene_const, case_direction = sim_env.initScene( reset_veh_list, RANDOMIZE, q_algo.course_eps )

        ###############
        # Miscellaneous
        ###############
        q_algo.updateMiscellaneous( global_step )

        # Print Rewards
        for v in reset_veh_list:
            print('========')
            print('Vehicle #:', v)
            print('\tGlobal Step     : ' + str(global_step))
            print('\tEPS             : ' + str(q_algo.eps))
            print('\tCourse EPS      : ' + str(q_algo.course_eps))
            print('\tEpisode #       : ' + str(epi_counter) + ' / ' + str(options.MAX_EPISODE) )
            print('\tStep            : ' + str(int(sim_env.epi_step_stack[v])) )
            print('\tEpisode Reward  : ' + str(sim_env.epi_reward_stack[v])) 
            # FIXME: case_direction tells the obstacle position. But scene is updated before printing this.
            # print('\tObs. Position   : ' + str(case_direction[v]) )        
            print('\tLast Loss       : ',data_package.avg_loss[-1])
            print('========')
            print('')

        # Update data
        for v in reset_veh_list:
            data_package.add_reward( sim_env.epi_reward_stack[v] )  # Add reward
            data_package.add_eps( q_algo.eps )                     # Add epsilon used for this reward
            data_package.add_success_rate( epi_sucess[v] )  # Add success/fail

        # Reset rewards for finished vehicles
        sim_env.resetRewards(veh_status)

        # save progress
        if options.TESTING == False:
            if options.NO_SAVE == False and epi_counter - last_saved_epi >= options.SAVER_RATE:
                q_algo.saveNetworkKeras(START_TIME_STR, epi_counter, global_step)
                print('-----------------------------------------')
                print("Saving data...") 
                print('-----------------------------------------')

                # Save Reward Data
                data_package.save_reward()
                data_package.save_loss()

                # Update variables
                last_saved_epi = epi_counter


        # GS_END_TIME_STR   = datetime.datetime.now()
        # print('Time : ' + str(GS_END_TIME_STR - GS_START_TIME_STR))

    # stop the simulation & close connection
    sim_env.end()

    ######################################################
    # Post Processing
    ######################################################

    # Print Current Time
    END_TIME = datetime.datetime.now()
    print("===============================")
    print("Start Time: ",START_TIME_STR)
    print("End Time  : ",str(END_TIME))
    print("Duration  : ",END_TIME-START_TIME)
    print("===============================")

    #############################3
    # Visualize Data
    ##############################3

    # Plot Reward
    data_package.plot_reward( options.RUNNING_AVG_STEP )
    data_package.plot_loss()


# END
