########
# rl_icm.py
#
# This file implements the estimation of forward dynamics. (This is only a partial implementation of the paper.)
#
# Specifically, no inverse dynamic estimation is implemented yet
#
#
########

import numpy as np
import tensorflow as tf
import math
import sys
import random

# Class for Neural Network
class ICM:
    def __init__(self, options, scene_const, name):
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
            # Input is [s_{t}, a_{t}]
            
            # Outputs
            # [\hat{s}_{t+1}]

            self.observation    = tf.placeholder(tf.float32, [None, (scene_const.sensor_count+2) + options.ACTION_DIM], name='icm_input')
            self.actual_state   = tf.placeholder(tf.float32, [None, (scene_const.sensor_count)+2], name='icm_target')

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

            # Sensor uses sigmoid since value varies from 0 to 1
            self.est_sensor = tf.layers.dense( inputs=self.h_s3,
                                              units=scene_const.sensor_count,
                                              activation = tf.nn.sigmoid,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name="est_sensor"
                                         )
        
            # Goal distance use sigmoid, goal angle use tanh
            self.est_goal_dist = tf.layers.dense( inputs=self.h_s3,
                                              units=1,
                                              activation = tf.nn.sigmoid,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name="est_g_dist"
                                         )
            self.est_goal_angle = tf.layers.dense( inputs=self.h_s3,
                                              units=1,
                                              activation = tf.nn.tanh,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              name="est_g_angle"
                                         )

            self.est_combined = tf.concat([self.est_sensor, self.est_goal_dist, self.est_goal_angle], -1)
            ######################################:
            ## END Constructing Neural Network
            ######################################:

            # Loss Scaling Factor. Scale Angle by 100. \sum (w_i x_i)^2
            loss_scale = np.ones( scene_const.sensor_count+2 )
            loss_scale[scene_const.sensor_count] = 100
            loss_scale = np.reshape(loss_scale, [-1, scene_const.sensor_count + 2] )
            # Loss for Optimization
            self.loss = tf.losses.mean_squared_error(
                                labels = self.actual_state,
                                predictions = self.est_combined,
                                weights = loss_scale
                            )
            self.loss = tf.reduce_mean(tf.square(self.actual_state - self.est_combined))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)

    # Evalulate the neural network to get the estimate of the next state
    def getEstimate(self, feed):
        return self.est_combined.eval(feed_dict = feed)


    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name
