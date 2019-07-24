import math
import random
import sys

import numpy as np
import tensorflow as tf
from icecream import ic

# Class for Neural Network
class QAgent:
    def __init__(self, options, scene_const, name):
        self.scope = name
        # Variables
        self.eps = options.INIT_EPS


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
            self.obs_sensor     = tf.placeholder(tf.float32, [None, scene_const.sensor_count, options.FRAME_COUNT], name='observation_sensor')
            self.obs_goal       = tf.placeholder(tf.float32, [None, 2, options.FRAME_COUNT], name='observation_goal')
            self.ISWeights      = tf.placeholder(tf.float32, [None,1], name='IS_weights')
            self.act            = tf.placeholder(tf.float32, [None, options.ACTION_DIM],name='action')
            self.target_Q       = tf.placeholder(tf.float32, [None, ], name='target_q' )  

            # self.obs_sensor_k   = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_sensor_k', batch_size = options.BATCH_SIZE )
            # self.obs_goal_k     = tf.keras.layers.Input( shape = ( 2, options.FRAME_COUNT), name='observation_goal_k', batch_size=options.BATCH_SIZE )
            self.obs_sensor_k   = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='observation_sensor_k')
            self.obs_goal_k     = tf.keras.layers.Input( shape = ( 2, options.FRAME_COUNT), name='observation_goal_k')
            # self.act_k          = tf.keras.layers.Input( shape = (options.ACTION_DIM,), name='action', batch_size=options.BATCH_SIZE )

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
                  inputs      = self.obs_sensor,
                  filters     = 10,
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

                self.h_s2 = tf.layers.flatten(
                  tf.layers.max_pooling1d(
                    inputs = 
                      tf.layers.conv1d(
                      inputs      = self.h_s1_max,
                      filters     = 20,
                      kernel_size = 3,
                      activation  = act_function,
                      padding     = 'valid',
                      kernel_initializer= tf.contrib.layers.xavier_initializer()
                    ),
                    pool_size = 2,
                    strides   = 2
                  )
                )

                self.h_flat = tf.layers.flatten( inputs = self.h_s2 )
                # self.h_s1 = tf.layers.dense( inputs=self.h_flat,
                #                              units=options.H1_SIZE,
                #                              activation = act_function,
                #                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                              kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=loss_scale),
                #                              name="h_s1"
                #                            )

                self.h_concat = tf.concat([self.h_flat, tf.layers.flatten( inputs = self.obs_goal )], -1)

                self.h_s2 = tf.layers.dense( inputs=self.h_concat,
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
                    print( tf.subtract(self.adv_est, tf.reduce_mean(self.adv_est, axis=1, keepdims=True) )  )
                    print( tf.reduce_mean(self.adv_est, axis=1, keepdims=True))

                    # Keras

                    # Typical Neural Net
                    network_structure = 2
                    if network_structure == 1:
                        self.h_flat1_k = tf.keras.layers.Flatten()(self.obs_goal_k)
                        self.h_flat2_k = tf.keras.layers.Flatten()(self.obs_sensor_k)
                        self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k])
                        self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation='relu')( self.h_concat_k )
                        self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation='relu')( self.h_dense1_k )
                        self.h_dense3_k = tf.keras.layers.Dense( options.H3_SIZE, activation='relu')( self.h_dense2_k )
                        self.h_out_k = tf.keras.layers.Dense( options.ACTION_DIM, activation=None, name='out_large')( self.h_dense3_k )
                    elif network_structure == 2:
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
                        self.h_s1_k   = tf.keras.layers.Conv1D(filters = 10, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu')(self.obs_sensor_k)
                        self.h_s1_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s1_k)
                        self.h_s2_k   = tf.keras.layers.Conv1D(filters = 20, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(self.h_s1_k_p)
                        self.h_s2_k_p = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)(self.h_s2_k)
                        self.h_flat   = tf.keras.layers.Flatten()(self.h_s2_k_p)
                        self.h_flat_g = tf.keras.layers.Flatten()(self.obs_goal_k)
                        self.h_concat_k = tf.keras.layers.concatenate([ self.h_flat, self.h_flat_g])
                        self.h_val = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_concat_k)
                        self.h_adv = tf.keras.layers.Dense( options.H3_SIZE, activation = 'relu')(self.h_concat_k)
                        self.h_val_est = tf.keras.layers.Dense( 1, activation = None)(self.h_val)
                        self.h_adv_est = tf.keras.layers.Dense( options.ACTION_DIM, activation = None)(self.h_adv)
                        # self.reduced_mean = tf.keras.backend.mean(self.h_adv_est, axis = 1, keepdims=True)
                        # print('Hi')
                        # print(tf.keras.layers.RepeatVector(5)(self.reduced_mean))
                        print(self.h_adv_est)
                        # self.sub = self.h_adv_est - self.reduced_mean
                        # self.h_out_k = self.h_val_est + self.sub
                        print( tf.keras.backend.mean( self.h_adv_est, axis = 1, keepdims=True) )

                        self.h_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.expand_dims(x[:,0], axis=1) + (x[:,1:] - tf.keras.backend.mean( x[:,1:], axis = 1, keepdims=True)), name = 'out_large'  )(tf.keras.layers.concatenate([self.h_val_est,self.h_adv_est]))

                    self.h_action_out_k = tf.keras.layers.Lambda( lambda x: tf.keras.backend.max( x, axis = 1, keepdims = True) )(self.h_out_k)

                    self.model = tf.keras.models.Model( inputs = [self.obs_goal_k, self.obs_sensor_k], outputs = self.h_action_out_k)

                    sgd = tf.keras.optimizers.Adam(lr = options.LR)
                    self.model.compile( optimizer=sgd,
                                        loss = 'mean_squared_error' 
                    )
                    self.model.summary()

                    # effectively computing twice to get value and maximum
                    self.model_out = tf.keras.Model(inputs = self.model.input, outputs = self.model.get_layer('out_large').output)
                    tf.keras.utils.plot_model( self.model, to_file='model2.png')
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
    def sample_action(self, feed, options):
        if random.random() <= self.eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = np.random.randint( options.ACTION_DIM, size=options.VEH_COUNT )
        else:
            act_values = self.output.eval(feed_dict=feed)
            if options.TESTING == True or options.VERBOSE == True:
                ic(np.argmax(act_values,axis=1))
                ic(act_values)
                print("\n")
                pass

            # Get maximum for each vehicle
            action_index = np.argmax(act_values, axis=1)

        return action_index

    def sample_action_k(self, feed, options):
        if random.random() <= self.eps and options.TESTING == False:             # pick random action if < eps AND testing disabled.
            # pick random action
            action_index = np.random.randint( options.ACTION_DIM, size=options.VEH_COUNT )
        else:
            act_values = self.model_out.predict(feed, batch_size=options.VEH_COUNT)
            if options.TESTING == True or options.VERBOSE == True:
                ic('Sample Action K')
                ic(np.argmax(act_values,axis=1))
                ic(act_values)
                print("\n")
                pass

            # Get maximum for each vehicle
            action_index = np.argmax(act_values, axis=1)

        return action_index

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name

    # update epsilon
    def decayEps( self, options, global_step ):
      # Decay epsilon
      if global_step % options.EPS_ANNEAL_STEPS == 0 and self.eps > options.FINAL_EPS:
          self.eps = self.eps * options.EPS_DECAY
