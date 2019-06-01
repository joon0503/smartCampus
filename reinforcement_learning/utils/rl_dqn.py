import numpy as np
import tensorflow as tf
import math
import sys
import random
from icecream import ic

# Class for Neural Network
class QAgent:
    def __init__(self, options, scene_const, name):
        self.scope = name
    
        # Parse input for activation function
        if options.ACT_FUNC == 'elu':
            act_function = tf.nn.elu
        elif options.ACT_FUNC == 'relu':
            act_function = tf.nn.relu
        else:
            raise NameError('Supplied activation function is not supported!')

        with tf.variable_scope(self.scope):      # Set variable scope

            loss_scale = options.L2_LOSS

            ######################################:
            ## CONSTRUCTING NEURAL NETWORK
            ######################################:

            # Inputs
#            self.sensor_input   = tf.placeholder(tf.float32, [None, scene_const.sensor_count*options.FRAME_COUNT], name='observation')
#            self.goal_input     = tf.placeholder(tf.float32, [None, 2*options.FRAME_COUNT], name='observation')
            self.observation    = tf.placeholder(tf.float32, [None, scene_const.sensor_count+2, options.FRAME_COUNT], name='observation')
            self.ISWeights      = tf.placeholder(tf.float32, [None,1], name='IS_weights')
            self.act            = tf.placeholder(tf.float32, [None, options.ACTION_DIM],name='action')
            self.target_Q       = tf.placeholder(tf.float32, [None, ], name='target_q' )  


            if False:
                # Slicing
                self.sensor_data    = tf.slice(self.observation, [0, 0], [-1, scene_const.sensor_count*options.FRAME_COUNT])
                self.goal_data      = tf.slice(self.observation, [0, scene_const.sensor_count*options.FRAME_COUNT], [-1, 2])

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
                                                 activation = None,
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

                    self.adv_est    = tf.layers.dense( inputs=self.h_layer_adv,
                                                 units=options.ACTION_DIM,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="adv_est"
                                               )
                 
                    self.output = self.val_est + tf.subtract(self.adv_est, tf.reduce_mean(self.adv_est, axis=1, keepdims=True) ) 
            else:
                # Regular neural net

                # Input is [BATCH_SIZE, sensor_count+2, frame_count]
                self.h_s1 = tf.layers.conv1d(
                  inputs      = self.observation,
                  filters     = 5,
                  kernel_size = 5,
                  activation  = act_function,
                  padding     = 'valid',
                  kernel_initializer= tf.contrib.layers.xavier_initializer()
                )
                # # Output is [BATCH_SIZE, sensor_count+2 - kernel_size = 21-3=18, filter_count = 5  ]

                self.h_s1_max = tf.layers.max_pooling1d(
                  inputs      = self.h_s1,
                  pool_size   = 2,
                  strides     = 2    
                )

                # self.h_s2 = tf.layers.flatten(
                #   tf.layers.max_pooling1d(
                #     inputs = 
                #       tf.layers.conv1d(
                #       inputs      = self.h_s1_max,
                #       filters     = 10,
                #       kernel_size = 3,
                #       activation  = act_function,
                #       padding     = 'valid',
                #       kernel_initializer= tf.contrib.layers.xavier_initializer()
                #     ),
                #     pool_size = 2,
                #     strides   = 2
                #   )
                # )

                self.h_flat = tf.layers.flatten( inputs = self.h_s1_max )
                # self.h_s1 = tf.layers.dense( inputs=self.h_flat,
                #                              units=options.H1_SIZE,
                #                              activation = act_function,
                #                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                              kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                #                              name="h_s1"
                #                            )
                self.h_s2 = tf.layers.dense( inputs=self.h_flat,
                                             units=options.H2_SIZE,
                                             activation = act_function,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                                             name="h_s2"
                                           )

                if options.disable_duel == True:
                    print("======================")
                    print("Regular Net")
                    print("======================")

                    self.h_s3 = tf.layers.dense( inputs=self.h_s2,
                                                 units=options.H3_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 name="h_s3"
                                               )
                    self.output = tf.layers.dense( inputs=self.h_s3,
                                                      units=options.ACTION_DIM,
                                                      activation = None,
                                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                      name="h_action"
                                                 )
                else:
                    print("======================")
                    print("Dueling Network")
                    print("======================")

                    # Dueling Network
                    self.h_layer_val = tf.layers.dense( inputs=self.h_s2,
                                                 units=options.H3_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                                                 name="h_layer_val"
                                               )
                    self.h_layer_adv = tf.layers.dense( inputs=self.h_s2,
                                                 units=options.H3_SIZE,
                                                 activation = act_function,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                                                 name="h_layer_adv"
                                               )

                    # Value and advantage estimate
                    self.val_est    = tf.layers.dense( inputs=self.h_layer_val,
                                                 units=1,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                                                 name="val_est"
                                               )

                    self.adv_est    = tf.layers.dense( inputs=self.h_layer_adv,
                                                 units=options.ACTION_DIM,
                                                 activation = None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
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
            self.loss = tf.reduce_mean(self.ISWeights * tf.square(self.target_Q - self.Q)) + tf.losses.get_regularization_loss()

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)


    # Sample action with random rate eps
    # output:
    #   action_index
    def sample_action(self, feed, eps, options):
        if random.random() <= eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = np.random.randint( options.ACTION_DIM, size=options.VEH_COUNT )
        else:
            act_values = self.output.eval(feed_dict=feed)
            if options.TESTING == True or options.VERBOSE == True:
                ic(np.argmax(act_values))
                ic(act_values)
                pass

            # Get maximum for each vehicle
            action_index = np.argmax(act_values, axis=1)

        return action_index

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name
