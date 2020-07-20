import math
import random
import sys

import numpy as np
import tensorflow as tf
from icecream import ic

# Class for actor
class Actor:
    def __init__(self, options, scene_const, name):
        # Inputs
        self.obs_sensor_k   = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_sensor_k')
        self.obs_state      = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_state')
        self.obs_goal_k     = tf.keras.layers.Input( shape = ( 2, options.FRAME_COUNT), name='observation_goal_k')

        # BATCH_SIZE x ACTION_DIM
        # self.action         = tf.keras.layers.Input( shape = (options.ACTION_DIM,), name='action_k')

        ########
        # KERAS
        ########

        # Typical Neural Net
        network_structure = 1
        if network_structure == 1:
            # Dense
            self.h_flat1_k = tf.keras.layers.Flatten()(self.obs_goal_k)
            self.h_flat2_k = tf.keras.layers.Flatten()(self.obs_sensor_k)
            self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k])
            self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation='relu')( self.h_concat_k )
            self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation='relu')( self.h_dense1_k )
            self.h_dense3_k = tf.keras.layers.Dense( options.H3_SIZE, activation='relu')( self.h_dense2_k )
            self.h_out_k = tf.keras.layers.Dense( options.ACTION_DIM, activation=None, name='out_large')( self.h_dense3_k )
        elif network_structure == 2:
            # Dense + dueling network
            self.h_flat1_k = tf.keras.layers.Flatten()(self.obs_goal_k)
            self.h_flat2_k = tf.keras.layers.Flatten()(self.obs_sensor_k)
            self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k])
            self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation='relu')( self.h_concat_k )
            self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation='relu')( self.h_dense1_k )
            self.h_val = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_dense2_k)
            self.h_adv = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_dense2_k)
            self.h_val_est = tf.keras.layers.Dense( 1, activation = None)(self.h_val)
            self.h_adv_est = tf.keras.layers.Dense( options.ACTION_DIM, activation = None)(self.h_adv)
            self.h_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.expand_dims(x[:,0], axis=1) + (x[:,1:] - tf.keras.backend.mean( x[:,1:], axis = 1, keepdims=True)), name = 'out_large'  )(tf.keras.layers.concatenate([self.h_val_est,self.h_adv_est]))
        else:
            # CNN + dueling network

            # CNN for sensor measurement
            self.h_s1_k   = tf.keras.layers.Conv1D(filters = 10, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')(self.obs_sensor_k)
            self.h_s1_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s1_k)
            self.h_s2_k   = tf.keras.layers.Conv1D(filters = 20, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(self.h_s1_k_p)
            self.h_s2_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s2_k)

            # CNN for detection state
            self.h_s1_d   = tf.keras.layers.Conv1D(filters = 10, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')(self.obs_state)
            self.h_s1_d_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s1_d)
            self.h_s2_d   = tf.keras.layers.Conv1D(filters = 20, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(self.h_s1_d_p)
            self.h_s2_d_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s2_d)

            self.h_flat   = tf.keras.layers.Flatten()(self.h_s2_k_p)
            self.h_flat_d = tf.keras.layers.Flatten()(self.h_s2_d_p)
            self.h_flat_g = tf.keras.layers.Flatten()(self.obs_goal_k)

            self.h_concat_k = tf.keras.layers.concatenate([ self.h_flat, self.h_flat_d, self.h_flat_g])
            # self.h_concat_k = tf.keras.layers.concatenate([ self.h_flat, self.h_flat_g])

            # 2 Layers of dense network before duleing part
            self.h_mid_1 = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_concat_k)
            self.h_mid_2 = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_1)

            self.h_val = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_2)
            self.h_adv = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_2)
            self.h_val_est = tf.keras.layers.Dense( 1, activation = None)(self.h_val)
            self.h_adv_est = tf.keras.layers.Dense( options.ACTION_DIM, activation = None)(self.h_adv)
            # self.reduced_mean = tf.keras.backend.mean(self.h_adv_est, axis = 1, keepdims=True)
            # print('Hi')
            # print(tf.keras.layers.RepeatVector(5)(self.reduced_mean))
            print(self.h_adv_est)
            # self.sub = self.h_adv_est - self.reduced_mean
            # self.h_out_k = self.h_val_est + self.sub
            print( tf.keras.backend.mean( self.h_adv_est, axis = 1, keepdims=True) )

            # Q-values
            self.h_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.expand_dims(x[:,0], axis=1) + (x[:,1:] - tf.keras.backend.mean( x[:,1:], axis = 1, keepdims=True)), name = 'out_large'  )(tf.keras.layers.concatenate([self.h_val_est,self.h_adv_est]))
        
        self.h_actor_out_all = tf.keras.layers.Dense( options.ACTION_DIM, activation='softmax', name = 'softmax_layer')(self.h_out_k)

        # Max Q-value
        # self.h_action_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.max( x, axis = 1, keepdims = True) )(self.h_out_k)
        # self.h_action_out_a = tf.keras.layers.dot([self.h_actor_out_all, self.action], axes=[1,1])

        self.model_p_all = tf.keras.models.Model( inputs = [self.obs_goal_k, self.obs_sensor_k, self.obs_state], outputs = self.h_actor_out_all)
        # self.model_pa = tf.keras.models.Model( inputs = [self.obs_goal_k, self.obs_sensor_k, self.obs_state, self.action], outputs = self.h_action_out_a)

        # keras_opt = tf.keras.optimizers.Adam(lr = options.LR, clipvalue = 10)
        keras_opt = tf.keras.optimizers.SGD(lr = options.LR)
        self.model_p_all.compile( optimizer= keras_opt,
                            loss = 'categorical_crossentropy' 
        )
        # self.model_pa.compile( optimizer= keras_opt,
        #                     loss = 'categorical_crossentropy' 
        # )
        ic('Actor p_all Summary')
        self.model_p_all.summary()
        # ic('Actor pa Summary')
        # self.model_pa.summary()

        # effectively computing twice to get value and maximum
        # self.model_q_all = tf.keras.Model(inputs = [self.obs_goal_k, self.obs_sensor_k, self.obs_state], outputs = self.model_qa.get_layer('out_large').output)

        # Figures
        tf.keras.utils.plot_model( self.model_p_all, to_file='actor_p_all.png')
        # tf.keras.utils.plot_model( self.model_pa, to_file='actor_pa.png')
        # tf.keras.utils.plot_model( self.model_q_all, to_file='model_q_all.png')

        return

    ######################################:
    ## END Constructing Neural Network
    ######################################:

    # Outputs
    # action_index : VEH_COUNT x 1 array, each index represent action applied to each vehicle. Action ranges from 0 ~ ACTION_DIM-1. 0 means left, ACTION_DIM means right
    def sample_action_k(self, feed, options):
        prob_values = self.model_p_all.predict(feed, batch_size=options.VEH_COUNT)
        # ic(prob_values, feed)
        # Get maximum for each vehicle
        action_index = np.zeros(options.VEH_COUNT, dtype=int)
        for i in range(0,options.VEH_COUNT):
            # FIXME: Modifying probability to ensure exploration. Currently, change minimum to 0.02 and apply softmax

            # Set minimum prob
            # modified_prob = np.maximum(prob_values[i,:], 0.02)

            # Adjust prob with maximum to sum to 1
            # modified_prob[np.argmax(prob_values[i,:])] = 1 + modified_prob[np.argmax(prob_values[i,:])] - np.sum(modified_prob)

            # ic(prob_values,modified_prob)
            action_index[i] = np.random.choice(options.ACTION_DIM, 1, p=prob_values[i,:])[0]

        if options.TESTING == True:
            ic(prob_values, action_index)
            print("\n")

        return action_index, prob_values

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name




# Class for Critique
class Critic:
    def __init__(self, options, scene_const, name):
        # Inputs
        self.obs_sensor_k   = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_sensor_k')
        self.obs_state      = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_state')
        self.obs_goal_k     = tf.keras.layers.Input( shape = ( 2, options.FRAME_COUNT), name='observation_goal_k')

        ########
        # KERAS
        ########

        # Typical Neural Net
        network_structure = 3
        if network_structure == 1:
            # Dense
            self.h_flat1_k = tf.keras.layers.Flatten()(self.obs_goal_k)
            self.h_flat2_k = tf.keras.layers.Flatten()(self.obs_sensor_k)
            self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k])
            self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation='relu')( self.h_concat_k )
            self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation='relu')( self.h_dense1_k )
            self.h_dense3_k = tf.keras.layers.Dense( options.H3_SIZE, activation='relu')( self.h_dense2_k )
            self.h_out_k = tf.keras.layers.Dense( options.ACTION_DIM, activation=None, name='out_large')( self.h_dense3_k )
        elif network_structure == 2:
            # Dense + dueling network
            self.h_flat1_k = tf.keras.layers.Flatten()(self.obs_goal_k)
            self.h_flat2_k = tf.keras.layers.Flatten()(self.obs_sensor_k)
            self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k])
            self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation='relu')( self.h_concat_k )
            self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation='relu')( self.h_dense1_k )
            self.h_val = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_dense2_k)
            self.h_adv = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_dense2_k)
            self.h_val_est = tf.keras.layers.Dense( 1, activation = None)(self.h_val)
            self.h_adv_est = tf.keras.layers.Dense( options.ACTION_DIM, activation = None)(self.h_adv)
            self.h_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.expand_dims(x[:,0], axis=1) + (x[:,1:] - tf.keras.backend.mean( x[:,1:], axis = 1, keepdims=True)), name = 'out_large'  )(tf.keras.layers.concatenate([self.h_val_est,self.h_adv_est]))
        else:
            # CNN + dueling network

            # CNN for sensor measurement
            self.h_s1_k   = tf.keras.layers.Conv1D(filters = 10, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')(self.obs_sensor_k)
            self.h_s1_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s1_k)
            self.h_s2_k   = tf.keras.layers.Conv1D(filters = 20, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(self.h_s1_k_p)
            self.h_s2_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s2_k)

            # CNN for detection state
            self.h_s1_d   = tf.keras.layers.Conv1D(filters = 10, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')(self.obs_state)
            self.h_s1_d_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s1_d)
            self.h_s2_d   = tf.keras.layers.Conv1D(filters = 20, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(self.h_s1_d_p)
            self.h_s2_d_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s2_d)

            self.h_flat   = tf.keras.layers.Flatten()(self.h_s2_k_p)
            self.h_flat_d = tf.keras.layers.Flatten()(self.h_s2_d_p)
            self.h_flat_g = tf.keras.layers.Flatten()(self.obs_goal_k)

            self.h_concat_k = tf.keras.layers.concatenate([ self.h_flat, self.h_flat_d, self.h_flat_g])
            # self.h_concat_k = tf.keras.layers.concatenate([ self.h_flat, self.h_flat_g])

            # 2 Layers of dense network before duleing part
            self.h_mid_1 = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_concat_k)
            self.h_mid_2 = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_1)

            self.h_val = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_2)
            # self.h_adv = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_mid_2)
            self.h_val_est = tf.keras.layers.Dense( 1, activation = None)(self.h_val)
            # self.h_adv_est = tf.keras.layers.Dense( options.ACTION_DIM, activation = None)(self.h_adv)
            # self.reduced_mean = tf.keras.backend.mean(self.h_adv_est, axis = 1, keepdims=True)
            # print('Hi')
            # print(tf.keras.layers.RepeatVector(5)(self.reduced_mean))
            # print(self.h_adv_est)
            # self.sub = self.h_adv_est - self.reduced_mean
            # self.h_out_k = self.h_val_est + self.sub
            # print( tf.keras.backend.mean( self.h_adv_est, axis = 1, keepdims=True) )

            # Q-values
            # self.h_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.expand_dims(x[:,0], axis=1) + (x[:,1:] - tf.keras.backend.mean( x[:,1:], axis = 1, keepdims=True)), name = 'out_large'  )(tf.keras.layers.concatenate([self.h_val_est,self.h_adv_est]))

        # Max Q-value
        # self.h_action_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.max( x, axis = 1, keepdims = True) )(self.h_out_k)
        # self.h_action_out_k = tf.keras.layers.dot([self.h_out_k, self.action], axes=[1,1])

        self.model_val = tf.keras.models.Model( inputs = [self.obs_goal_k, self.obs_sensor_k, self.obs_state], outputs = self.h_val_est)
        # self.model_qa = tf.keras.models.Model( inputs = [self.obs_goal_k, self.obs_sensor_k, self.action], outputs = self.h_action_out_k)

        # keras_opt = tf.keras.optimizers.Adam(lr = options.LR, clipvalue = 10)
        keras_opt = tf.keras.optimizers.Adam(lr = options.LR)
        self.model_val.compile( optimizer= keras_opt,
                            loss = 'mean_squared_error' 
        )
        ic('Critic Summary')
        self.model_val.summary()

        # effectively computing twice to get value and maximum
        # self.model_q_all = tf.keras.Model(inputs = [self.obs_goal_k, self.obs_sensor_k, self.obs_state], outputs = self.model_qa.get_layer('out_large').output)

        # Figures
        tf.keras.utils.plot_model( self.model_val, to_file='model_critic.png')
        # tf.keras.utils.plot_model( self.model_q_all, to_file='model_q_all.png')

        return

    ######################################:
    ## END Constructing Neural Network
    ######################################:

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name