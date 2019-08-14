#####################################
# q_algorithm.py
#
# This file contains the class for q_algorithm. 
# This uses QAgent class and replay memory among other things to implement the overall algorith,
#####################################
import numpy as np
import time
import os
from utils.rl_dqn import QAgent
from utils.experience_replay import Memory


class dqn:
    def __init__(self, sim_env):

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


        return


    # update epsilon
    def decayEps( self, options, global_step ):
        # Decay epsilon
        if global_step % options.EPS_ANNEAL_STEPS == 0 and self.eps > options.FINAL_EPS:
            self.eps = self.eps * options.EPS_DECAY

        return


    # Get optimal action Keras. 
    def getOptimalAction( self, obs_sensor_stack, obs_goal_stack ):
        action_feed = {}
        action_feed.clear()
        action_feed.update({'observation_sensor_k': obs_sensor_stack})
        action_feed.update({'observation_goal_k': obs_goal_stack})

        action_stack_k = agent_train.sample_action_k( action_feed, self.options )

        # Apply the Steering Action & Keep Velocity. For some reason, +ve means left, -ve means right
        # targetSteer = sim_env.scene_const.max_steer - action_stack * abs(sim_env.scene_const.max_steer - sim_env.scene_const.min_steer)/(options.ACTION_DIM-1)
        targetSteer_k = self.scene_const.max_steer - action_stack_k * abs(self.scene_const.max_steer - self.scene_const.min_steer)/(self.options.ACTION_DIM-1)

        return targetSteer_k

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

        # actions converted to value array
        #actions_mb_val = oneHot2Angle( actions_mb_hot, sim_env.scene_const, options, radians = False, scale = True )

        # ic(states_mb,actions_mb,rewards_mb,next_states_mb,done_mb)
        # ic( next_states_mb.reshape(-1,sim_env.scene_const.sensor_count+2,options.FRAME_COUNT) )

        # Get Target Q-Value
        # feed.clear()
        # feed.update({agent_train.obs_sensor : next_states_sensor_mb, agent_train.obs_goal : next_states_goal_mb})

        # Calculate Target Q-value. Uses double network. First, get action from training network
        # action_train = np.argmax( agent_train.output.eval(feed_dict=feed), axis=1 )
        self.action_train = self.agent_train.model_out.predict(
                                            {
                                                'observation_sensor_k' : next_states_sensor_mb,
                                                'observation_goal_k'   : next_states_goal_mb
                                            },
                                            batch_size = self.options.VEH_COUNT
        )
        action_train_k = np.argmax( action_train_k, axis=1)
        # ic(np.argmax(action_train_k,axis=1))

        if options.disable_DN == False:
            # feed.clear()
            # feed.update({agent_target.obs_sensor : next_states_sensor_mb, agent_target.obs_goal : next_states_goal_mb})

            keras_feed = {}
            keras_feed.clear()
            keras_feed.update(
                {
                    'observation_sensor_k' : next_states_sensor_mb,
                    'observation_goal_k'   : next_states_goal_mb
                }

            )
            # Using Target + Double network
            # ic( agent_target.output.eval( feed_dict = feed), agent_target.output.eval( feed_dict = feed).shape  )
            # ic( agent_target.h_s1.eval( feed_dict = feed), agent_target.h_s1.eval( feed_dict = feed).shape  )
            # ic( agent_target.output.eval(feed_dict=feed)[np.arange(0,options.BATCH_SIZE),action_train] )
            # q_target_val = rewards_mb + options.GAMMA * agent_target.output.eval(feed_dict=feed)[np.arange(0,options.BATCH_SIZE),action_train]
            q_target_val_k = rewards_mb + options.GAMMA * self.agent_target.model_out.predict(keras_feed)[np.arange(0,self.options.BATCH_SIZE),action_train_k]
        else:
            keras_feed.clear()
            keras_feed.update(
                {
                    'observation_sensor_k' : next_states_sensor_mb,
                    'observation_goal_k'   : next_states_goal_mb
                }

            )
            # Just using Target Network.
            q_target_val = rewards_mb + options.GAMMA * np.amax( agent_target.output.eval(feed_dict=feed), axis=1)
            q_target_val_k = rewards_mb + options.GAMMA * np.amas( agent_target.model.predict(keras_feed), axis=1)
    
        # set q_target to reward if episode is done
        for v_mb in range(0,options.BATCH_SIZE):
            if done_mb[v_mb] == 1:
                # q_target_val[v_mb] = rewards_mb[v_mb]
                q_target_val_k[v_mb] = rewards_mb[v_mb]

        # Train Keras Model
        keras_feed = {}
        keras_feed.clear()
        keras_feed.update({ 'observation_sensor_k' : states_sensor_mb, 'observation_goal_k' : states_goal_mb})
        loss_k = self.agent_train.model.train_on_batch( keras_feed, np.reshape(q_target_val_k,(options.BATCH_SIZE,1)) )

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
