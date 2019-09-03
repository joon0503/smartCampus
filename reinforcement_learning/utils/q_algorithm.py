#####################################
# q_algorithm.py
#
# This file contains the class for q_algorithm. 
# This uses QAgent class and replay memory among other things to implement the overall algorith,
#####################################
import os
import time
import warnings

import numpy as np

from utils.experience_replay import Memory
from utils.rl_dqn import QAgent


class dqn:
    def __init__(self, sim_env, load = True):

        # Target and Training agent
        self.agent_train     = QAgent(sim_env.options,sim_env.scene_const, 'Training')
        self.agent_target    = QAgent(sim_env.options,sim_env.scene_const, 'Target')

        # Decay Step
        self.eps = sim_env.options.INIT_EPS

        # Options and Scene Consts
        self.options = sim_env.options
        self.scene_const = sim_env.scene_const

        # The replay memory.
        if sim_env.options.enable_PER == False:
            # Don't use PER
            print("=================================================")
            print("NOT using PER!")
            print("=================================================")
            self.replay_memory = Memory(sim_env.options.MAX_EXPERIENCE)
        else:
            # Use PER
            print("=================================================")
            print("Using PER!")
            print("=================================================")
            self.replay_memory = Memory(sim_env.options.MAX_EXPERIENCE, disable_PER = False, absolute_error_upperbound = 2000)

        if self.options.NO_SAVE == False and load == True:
            self.loadNetwork()
        else:
            warnings.warn("Warning: Loading network weight may have failed!")

        return

    # update miscellaneous
    # For now
    #   1. decay eps
    #   2. update target
    def updateMiscellaneous( self, global_step ):
        # Decay epsilon
        self.__decayEps( global_step )

        # Update target
        self.__updateTarget( global_step )

        return

    # Update target
    def __updateTarget(self, global_step):
        # Update target network
        if global_step % self.options.TARGET_UPDATE_STEP == 0:
            print('-----------------------------------------')
            print("Updating Target network.")
            print('-----------------------------------------')
            self.agent_target.model.set_weights( self.agent_train.model.get_weights() )

        return

    # update epsilon
    def __decayEps( self, global_step ):
        # Decay epsilon
        if global_step % self.options.EPS_ANNEAL_STEPS == 0 and self.eps > self.options.FINAL_EPS:
            self.eps = self.eps * self.options.EPS_DECAY

        return


    # Get optimal action Keras. 
    #   action_feed: dictionary of input to keras
    def getOptimalAction( self, action_feed ):
        action_stack_k = self.agent_train.sample_action_k( action_feed, self.eps, self.options )

        # Apply the Steering Action & Keep Velocity. For some reason, +ve means left, -ve means right
        # targetSteer = sim_env.scene_const.max_steer - action_stack * abs(sim_env.scene_const.max_steer - sim_env.scene_const.min_steer)/(options.ACTION_DIM-1)
        targetSteer_k = self.scene_const.max_steer - action_stack_k * abs(self.scene_const.max_steer - self.scene_const.min_steer)/(self.options.ACTION_DIM-1)

        return targetSteer_k, action_stack_k

    def trainOneStep( self ):
        # Obtain the mini batch. (Batch Memory is '2D array' with BATCH_SIZE X size(experience)
        tree_idx, batch_memory, ISWeights_mb = self.replay_memory.sample(self.options.BATCH_SIZE)

        # Get state/action/next state from obtained memory. Size same as queues
        states_sensor_mb        = np.array([each[0][0] for each in batch_memory])           # BATCH_SIZE x SENSOR_COUNT
        states_goal_mb          = np.array([each[0][1] for each in batch_memory])           # BATCH_SIZE x 2
        actions_mb              = np.array([each[0][2] for each in batch_memory])           # BATCH_SIZE x ACTION_DIM
        rewards_mb              = np.array([each[0][3] for each in batch_memory])           # 1 x BATCH_SIZE
        next_states_sensor_mb   = np.array([each[0][4] for each in batch_memory])   
        next_states_goal_mb     = np.array([each[0][5] for each in batch_memory])   
        done_mb                 = np.array([each[0][6] for each in batch_memory])   

        # actions mb is list of numbers. Need to change it into one hot encoding
        actions_mb_hot = np.zeros((self.options.BATCH_SIZE,self.options.ACTION_DIM))
        actions_mb_hot[np.arange(self.options.BATCH_SIZE),np.asarray(actions_mb, dtype=int)] = 1

        # Calculate Target Q-value. Uses double network. First, get action from training network
        action_train_k = self.agent_train.model_out.predict(
                                            {
                                                'observation_sensor_k' : next_states_sensor_mb,
                                                'observation_goal_k'   : next_states_goal_mb
                                            },
                                            batch_size = self.options.VEH_COUNT
        )
        action_train_k = np.argmax( action_train_k, axis=1)

        if self.options.disable_DN == False:
            keras_feed = {}
            keras_feed.clear()
            keras_feed.update(
                {
                    'observation_sensor_k' : next_states_sensor_mb,
                    'observation_goal_k'   : next_states_goal_mb
                }

            )
            # Using Target + Double network
            q_target_val_k = rewards_mb + self.options.GAMMA * self.agent_target.model_out.predict(keras_feed)[np.arange(0,self.options.BATCH_SIZE),action_train_k]
    
        # set q_target to reward if episode is done
        for v_mb in range(0,self.options.BATCH_SIZE):
            if done_mb[v_mb] == 1:
                q_target_val_k[v_mb] = rewards_mb[v_mb]

        # Train Keras Model
        keras_feed = {}
        keras_feed.clear()
        keras_feed.update({ 'observation_sensor_k' : states_sensor_mb, 'observation_goal_k' : states_goal_mb})

        # Loss
        loss_k = self.agent_train.model.train_on_batch( keras_feed, np.reshape(q_target_val_k,(self.options.BATCH_SIZE,1)) )

        return loss_k, states_sensor_mb, next_states_sensor_mb, states_goal_mb, next_states_goal_mb, actions_mb

    # Load network wegiths
    def loadNetwork( self ):
        weight_path = None

        # If file is provided
        if self.options.WEIGHT_FILE != None:
            weight_path = self.options.WEIGHT_FILE
        elif os.path.isfile('./checkpoints-vehicle/checkpoint.txt'):
            with open('./checkpoints-vehicle/checkpoint.txt') as check_file:
                weight_file = check_file.readline()
                weight_path = './checkpoints-vehicle/' + weight_file 

        if weight_path == None:
            print("\n\n=================================================")
            print("=================================================")
            print("Could not find old network weights")
            print("=================================================")
            print("=================================================\n\n")
        else:
            self.agent_train.model.load_weights( weight_path )
            self.agent_target.model.load_weights( weight_path )
            print("\n\n=================================================")
            print("=================================================")
            print("Successfully loaded:", weight_path)
            print("=================================================")
            print("=================================================\n\n")

        time.sleep(2)
        return

    # Save network weights
    # Inputs
    #   START_TIME_STR : string of starting time
    #   epi_counter    : current episode number
    #   global_step    : current global step
    def saveNetworkKeras(self, START_TIME_STR, epi_counter, global_step):
        print('-----------------------------------------')
        print("Saving network...")
        print('-----------------------------------------')
        if not os.path.exists('./checkpoints-vehicle'):
            os.makedirs('./checkpoints-vehicle')
        self.agent_train.model.save_weights('./checkpoints-vehicle/' + START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '.h5', overwrite=True)

        # Save checkpoint
        with open('./checkpoints-vehicle/checkpoint.txt','w') as check_file:
            check_file.write(START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '.h5')

        return
