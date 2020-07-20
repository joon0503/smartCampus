#####################################
# a2c_algorithm.py
#
# This file contains the class for a2c_algorithm. 
#####################################
import os
import time
import warnings
from icecream import ic

import numpy as np

from utils.experience_replay import Memory
# from utils.a2c_actor_critic_class import QAgent
from utils.a2c_actor_critic_class import Actor
from utils.a2c_actor_critic_class import Critic


class a2c:
    def __init__(self, sim_env, init_course_eps = 1.0, load = True):
        # Target and Training agent
        self.agent_actor      = Actor(sim_env.options,sim_env.scene_const, 'Actor')
        self.agent_critic     = Critic(sim_env.options,sim_env.scene_const, 'Critic')
        # self.agent_target    = QAgent(sim_env.options,sim_env.scene_const, 'Target')

        # Decay Step
        self.eps = sim_env.options.INIT_EPS

        # Options and Scene Consts
        self.options = sim_env.options
        self.scene_const = sim_env.scene_const
        self.sim_env = sim_env

        # Course eps
        self.course_eps = init_course_eps

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
        # self.__updateTarget( global_step )

        # Increase course hardness
        self.__updateCourseEps( global_step )
        return

    # Update target
    # def __updateTarget(self, global_step):
    #     # Update target network
    #     if global_step % self.options.TARGET_UPDATE_STEP == 0:
    #         print('-----------------------------------------')
    #         print("Updating Target network.")
    #         print('-----------------------------------------')
    #         self.agent_target.model_qa.set_weights( self.agent_train.model_qa.get_weights() )

    #     return

    # update epsilon
    def __decayEps( self, global_step ):
        # Decay epsilon
        if global_step % self.options.EPS_ANNEAL_STEPS == 0 and self.eps > self.options.FINAL_EPS:
            self.eps = self.eps * self.options.EPS_DECAY

        return

    # update course eps
    def __updateCourseEps( self, global_step ):
        # Decay epsilon
        if global_step % self.options.EPS_ANNEAL_STEPS == 0:
            self.course_eps = self.course_eps * self.options.EPS_DECAY

        # Jump the obstacle if too close
        if abs(self.course_eps - 0.75) < 0.01:
            self.course_eps = 0.5

        return

    # Get optimal action Keras. 
    #   action_feed: dictionary of input to keras
    #   all: print out probabilities as well
    def getOptimalAction( self, action_feed ):
        action_stack_k, action_prob = self.agent_actor.sample_action_k( action_feed, self.options )

        # Apply the Steering Action & Keep Velocity. For some reason, +ve means left, -ve means right
        # Recall action_stack_k is index with 0 means left, and ACTION_DIM-1 mean right.
        # Hence, if action_index = 0, targetSteer = max_steer and action_index=ACTION_DIM-1 means targetSteer = -1*max_steer
        # targetSteer = sim_env.scene_const.max_steer - action_stack * abs(sim_env.scene_const.max_steer - sim_env.scene_const.min_steer)/(options.ACTION_DIM-1)
        targetSteer_k = self.scene_const.max_steer - action_stack_k * abs(self.scene_const.max_steer - self.scene_const.min_steer)/(self.options.ACTION_DIM-1)

        return targetSteer_k, action_stack_k, action_prob

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

        # Calculate current & next value
        curr_value = self.agent_critic.model_val.predict(
                                            {
                                                'observation_sensor_k' : states_sensor_mb[:,0:self.scene_const.sensor_count,:],
                                                'observation_state'    : states_sensor_mb[:,self.scene_const.sensor_count:,:],
                                                'observation_goal_k'   : states_goal_mb
                                            },
                                            batch_size = self.options.VEH_COUNT
        )

        next_value = self.agent_critic.model_val.predict(
                                            {
                                                'observation_sensor_k' : next_states_sensor_mb[:,0:self.scene_const.sensor_count,:],
                                                'observation_state'    : next_states_sensor_mb[:,self.scene_const.sensor_count:,:],
                                                'observation_goal_k'   : next_states_goal_mb
                                            },
                                            batch_size = self.options.VEH_COUNT
        )

        # Find Value Target
        value_target = np.reshape(rewards_mb, (self.options.BATCH_SIZE,1)) + self.options.GAMMA * next_value


        # Find Advantage
        advantage = np.zeros((self.options.BATCH_SIZE, self.options.ACTION_DIM))
        for i in range(0,self.options.BATCH_SIZE): 
            advantage[i,actions_mb[i]] = rewards_mb[i] + self.options.GAMMA * next_value[i] - curr_value[i]
        
        # Get action from training model
        # action_train_k = np.argmax( q_val_train, axis=1)
        # action_k = self.agent_actor.sample_action_k

        # keras_feed = {}
        # keras_feed.clear()
        # keras_feed.update(
        #     {
        #         'observation_sensor_k' : next_states_sensor_mb[:,0:self.scene_const.sensor_count,:],
        #         'observation_state'    : next_states_sensor_mb[:,self.scene_const.sensor_count:,:],
        #         'observation_goal_k'   : next_states_goal_mb
        #     }

        # )
        # Using Target + Double network
        # q_target_val_vec = rewards_mb + self.options.GAMMA * self.agent_target.model_q_all.predict(keras_feed)[np.arange(0,self.options.BATCH_SIZE),action_train_k]

        # set q_target to reward if episode is done
        # for v_mb in range(0,self.options.BATCH_SIZE):
        #     if done_mb[v_mb] == 1:
        #         q_target_val_vec[v_mb] = rewards_mb[v_mb]

        for v_mb in range(0,self.options.BATCH_SIZE):
            if done_mb[v_mb] == 1:
                advantage[v_mb,actions_mb[v_mb]] = rewards_mb[v_mb] - curr_value[v_mb]
                value_target[v_mb] = rewards_mb[v_mb]

        # Target is BATCH_SIZE x ACTION_DIM, with Q(s_t,a_t) replaced with \hat{Q}
        # q_target_val_mtx = q_val_train
        # q_target_val_mtx[ np.arange(self.options.BATCH_SIZE), action_train_k] = q_target_val_vec


        # ic(curr_value, next_value, value_target,advantage)

        # Train Keras Model
        keras_feed = {}
        keras_feed.clear()
        keras_feed.update(
            { 
                'observation_sensor_k' : states_sensor_mb[:,0:self.scene_const.sensor_count,:], 
                'observation_state'    : states_sensor_mb[:,self.scene_const.sensor_count:,:], 
                'observation_goal_k'   : states_goal_mb,
                # 'action_k'             : actions_mb_hot
            }
        )

        if self.options.VERBOSE == True:
            ic(keras_feed)
            # ic( np.reshape(q_target_val_mtx,(self.options.BATCH_SIZE,self.options.ACTION_DIM)) )

        # if self.options.TESTING == True:
            # ic(advantage)

        # FIXME: using -1*advantage as the target??
        # Loss
        loss_actor  = self.agent_actor.model_p_all.train_on_batch( keras_feed, advantage)        # Data, Target
        loss_critic = self.agent_critic.model_val.train_on_batch( keras_feed, np.reshape(value_target,(self.options.BATCH_SIZE,1)) )
        # loss_k = self.agent_train.model_qa.train_on_batch( keras_feed, np.reshape(q_target_val_vec,(self.options.BATCH_SIZE,1)) )

        return [loss_actor, loss_critic], states_sensor_mb, next_states_sensor_mb, states_goal_mb, next_states_goal_mb, actions_mb

    # Load network wegiths
    def loadNetwork( self ):
        weight_path = None

        # If file is provided. From checkpoint.txt, read the last line, i.e., the latest weight file.
        # For a2c, weight files does NOT include extension, and indicator for agent and critic
        if self.options.WEIGHT_FILE != None:
            weight_path_actor = './a2c-checkpoints-vehicle/' + self.options.WEIGHT_FILE + '_actor.h5'
            weight_path_critic = './a2c-checkpoints-vehicle/' + self.options.WEIGHT_FILE + '_critic.h5'

            ic(weight_path_actor, weight_path_critic)
        else:
            if os.path.isfile('./a2c-checkpoints-vehicle/checkpoint_actor.txt'):
                with open('./a2c-checkpoints-vehicle/checkpoint_actor.txt') as check_file:
                    # Get file content
                    weight_file = check_file.readlines()

                    # Get last line
                    weight_file = weight_file[len(weight_file)-1].rstrip()

                    weight_path_actor = './a2c-checkpoints-vehicle/' + weight_file 

                    ic(weight_path_actor)

                with open('./a2c-checkpoints-vehicle/checkpoint_critic.txt') as check_file:
                    # Get file content
                    weight_file = check_file.readlines()

                    # Get last line
                    weight_file = weight_file[len(weight_file)-1].rstrip()

                    weight_path_critic = './a2c-checkpoints-vehicle/' + weight_file 

                    ic(weight_path_critic)
            else:
                weight_path_actor = None
                weight_path_critic = None
        if weight_path_actor == None:
            print("\n\n=================================================")
            print("=================================================")
            print("Could not find old network weights")
            print("=================================================")
            print("=================================================\n\n")
        else:
            self.agent_actor.model_p_all.load_weights( weight_path_actor )
            self.agent_critic.model_val.load_weights( weight_path_critic )
            print("\n\n=================================================")
            print("=================================================")
            print("Successfully loaded:", weight_path_actor, weight_path_critic)
            print("=================================================")
            print("=================================================\n\n")

        time.sleep(1)
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
        if not os.path.exists('./a2c-checkpoints-vehicle'):
            os.makedirs('./a2c-checkpoints-vehicle')
        # self.agent_train.model.save_weights('./checkpoints-vehicle/' + START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '.h5', overwrite=True)
        self.agent_actor.model_p_all.save('./a2c-checkpoints-vehicle/' + START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '_actor.h5', overwrite=True)
        self.agent_critic.model_val.save('./a2c-checkpoints-vehicle/' + START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '_critic.h5', overwrite=True)

        # FIXME: Saving checkpoint not working correctly
        # Save checkpoint
        with open('./a2c-checkpoints-vehicle/checkpoint_actor.txt','a+') as check_file:
            check_file.write(START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '_actor.h5\n')

        with open('./a2c-checkpoints-vehicle/checkpoint_critic.txt','a+') as check_file:
            check_file.write(START_TIME_STR + "_e" + str(epi_counter) + "_gs" + str(global_step) + '_critic.h5\n')

        return

