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
import matplotlib.pyplot as plt


# Class for Neural Network
class ICM:
    def __init__(self, options, scene_const, name):
        # General Variables for plotting
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.data_x = []
        self.data_y = []
        self.data_arrow_x = []
        self.data_arrow_y = []

        ########################
        # NEURAL NET
        ########################
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
            loss_scale = np.ones( scene_const.sensor_count+2 )*10
            loss_scale[scene_const.sensor_count] = 10e4                                 # error for angle
            loss_scale = np.reshape(loss_scale, [-1, scene_const.sensor_count + 2] )

            # Loss for Optimization
            self.loss = tf.losses.mean_squared_error(
                                labels = self.actual_state,
                                predictions = self.est_combined,
                                weights = loss_scale
                            )
            #self.loss = tf.reduce_mean(tf.square(self.actual_state - self.est_combined))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(options.LR).minimize(self.loss)

    # Evalulate the neural network to get the estimate of the next state
    def getEstimate(self, feed):
        return self.est_combined.eval(feed_dict = feed)


    # Plot current position of the vehicle and the estimation   
    #   curr_state : [sensor_values, goal_angle, goal_distance]
    #   action     : one hot encoded vector
    #   veh_heading: headings of vehicle gamma in radians. 
    #                we use gamma and 0 deg -> front, +90 deg -> right, -90 deg -> left
    #   scene_const: scene constants as the class

    def plotEstimate(self, curr_state, action, veh_heading, scene_const):
        ####################
        # Proccess Data
        ####################
        veh_x, veh_y, arrow_x, arrow_y = self.getPoints(curr_state,action,scene_const)

        # Clear data if close to start line and data is short. FIXME: Currently, does not reset if collision occurs immediately
        self.ax.clear()
        if len(self.data_y) > 0 and abs(self.data_y[-1] - veh_y) > 5:
            self.data_x = []
            self.data_y = []
            self.data_arrow_x = []
            self.data_arrow_y = []

        # Update data array
        self.data_x.append(veh_x) 
        self.data_y.append(veh_y) 
        self.data_arrow_x.append(arrow_x) 
        self.data_arrow_y.append(arrow_y) 
       
        # Radar positions
        radar_x = []
        radar_y = []
        radar_x_col = []            # Radius where collision occurs
        radar_y_col = []

        if True:        
            # RELATIVE
            veh_heading = 0
        
        for i in range(0,scene_const.sensor_count):
            # gamma + sensor_min_angle   is the current angle of the left most sensor
            radar_x.append( scene_const.sensor_distance*curr_state[i]*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + i*scene_const.sensor_delta ) + veh_x )
            radar_y.append( scene_const.sensor_distance*curr_state[i]*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + i*scene_const.sensor_delta ) + veh_y )

            radar_x_col.append( scene_const.collision_distance*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + i*scene_const.sensor_delta ) + veh_x )
            radar_y_col.append( scene_const.collision_distance*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + i*scene_const.sensor_delta ) + veh_y )

 
        ####################
        # Get Estimate 
        ####################
        max_horizon = 1
        state_estimate_x = []
        state_estimate_y = []
        radar_est_x = []
        radar_est_y = []
        radar_est_x_col = []
        radar_est_y_col = []
        for i in range(0,max_horizon):
            temp_state = self.getEstimate( {self.observation : np.reshape(np.concatenate([curr_state, action]), [-1, (scene_const.sensor_count+2) + 5])  }  )
            temp_state = np.reshape(temp_state, -1)
            temp_x, temp_y, _, _ = self.getPoints( temp_state, action, scene_const)
            state_estimate_x.append(temp_x)
            state_estimate_y.append(temp_y)

            for k in range(0,scene_const.sensor_count):
                radar_est_x.append( scene_const.sensor_distance*temp_state[k]*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_x )
                radar_est_y.append( scene_const.sensor_distance*temp_state[k]*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_y )

                radar_est_x_col.append( scene_const.collision_distance*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_x )
                radar_est_y_col.append( scene_const.collision_distance*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_y )



        ####################
        # Plot
        ####################
        
        # Goal Point
        self.ax.scatter(0,60, color='blue')

        if True:
            # Scatter

            # Vehicle Trajectory & Input
            self.ax.scatter(self.data_x,self.data_y, color='red')

            # quiver the vehicle heading
            #self.ax.quiver(veh_x,veh_y, np.sin(veh_heading*math.pi), np.cos(veh_heading*math.pi), color='red')

            # Estimate
            self.ax.scatter(state_estimate_x,state_estimate_y, color='green')
        else:
            # Also plot input
            self.ax.quiver(self.data_x,self.data_y,self.data_arrow_x,self.data_arrow_y, color='red')

        # Radar
        self.ax.scatter(radar_x,radar_y, color='red', marker = 'x')
        self.ax.plot(radar_x,radar_y, color='red', label = 'True')

        # Collision
        #self.ax.scatter(radar_x_col,radar_y_col, color='red', marker = 'x')
        self.ax.plot(radar_x_col,radar_y_col, color='red', label = 'Collision Range', linestyle = '--')
        self.ax.plot(radar_est_x_col,radar_est_y_col, color='green', label = 'Collision Range (Est.)', linestyle = '--')

        # Radar Estimate
        #print('radar_est_x', radar_est_x)
        #print('radar_est_y', radar_est_y)
        self.ax.scatter(radar_est_x,radar_est_y, color='green', marker = 'x')
        self.ax.plot(radar_est_x,radar_est_y, color='green', label = 'Estimate')
 
        # Figure Properties
        self.ax.set_xlim(-6,6) 
        self.ax.set_ylim(0,60) 
        self.ax.legend()

        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        return

    # Compute data required for plotting from state and action values
    # Input
    #   curr_state : [sensor_values, goal_angle, goal_distance]
    #   action     : one hot encoded vector
    #   scene_const: scene constants as the class
    #
    # Output
    #   veh_x, veh_y: x,y coordinate of vehicle. 
    #   arrow_x, arrow_y: x,y coordinate of arrow computed from input

    def getPoints(self, curr_state, action, scene_const):
        #print(curr_state[9])
        # Break up into raw data
        raw_sensor = curr_state[0:scene_const.sensor_count]
        raw_angle  = curr_state[scene_const.sensor_count]
        raw_dist   = curr_state[scene_const.sensor_count+1]

        # Get position of vehicle from goal point distance and angle from raw data
        veh_x = -1*scene_const.goal_distance * raw_dist * np.sin( math.pi * raw_angle )         # -1 since positive distance means goal is right of vehicle. Since goal is at x=0, from goal's view, vehicle is at left of it, meaning negative.
        veh_y = scene_const.goal_distance - scene_const.goal_distance * raw_dist * np.cos( math.pi * raw_angle )

        # Get arrow
        # Delta of angle between each action
        action_delta = (scene_const.max_steer - scene_const.min_steer) / (5-1)

        # Calculate desired angle
        desired_angle = scene_const.max_steer - np.argmax(action) * action_delta
        arrow_x = 0.1*np.sin( math.radians( desired_angle ) )
        arrow_y = 0.1*np.cos( math.radians( desired_angle ) )

        return veh_x, veh_y, arrow_x, arrow_y

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name
