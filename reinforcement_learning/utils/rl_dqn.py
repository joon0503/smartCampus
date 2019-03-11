import numpy as np
import tensorflow as tf
import math
import sys
import random

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
            return 

        with tf.variable_scope(self.scope):      # Set variable scope

            ######################################:
            ## CONSTRUCTING NEURAL NETWORK
            ######################################:

            # Inputs
#            self.sensor_input   = tf.placeholder(tf.float32, [None, scene_const.sensor_count*options.FRAME_COUNT], name='observation')
#            self.goal_input     = tf.placeholder(tf.float32, [None, 2*options.FRAME_COUNT], name='observation')
            self.observation    = tf.placeholder(tf.float32, [None, (scene_const.sensor_count+2)*options.FRAME_COUNT], name='observation')
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
                                                 name="h_layer_val"
                                               )
                    self.h_layer_adv = tf.layers.dense( inputs=self.h_s2,
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
            ######################################:
            ## END Constructing Neural Network
            ######################################:

            # Prediction of given specific action
            self.Q = tf.reduce_sum( tf.multiply(self.output, self.act), axis=1)

            # Absolute Loss
            self.loss_per_data = tf.abs(self.target_Q - self.Q, name='loss_per_data')    
            
            # Loss for Optimization
            self.loss = tf.reduce_mean(self.ISWeights * tf.square(self.target_Q - self.Q))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)


    # Sample action with random rate eps
    def sample_action(self, feed, eps, options, verbose = False):
        if random.random() <= eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = random.randrange(options.ACTION_DIM)
#            action = random.uniform(0,1)
        else:
            act_values = self.output.eval(feed_dict=feed)
            if options.TESTING == True or verbose == True:
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
