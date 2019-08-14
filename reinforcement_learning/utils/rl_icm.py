########
# rl_icm.py
#
# This file implements the estimation of forward dynamics. (This is only a partial implementation of the paper.)
#
# Specifically, no inverse dynamic estimation is implemented yet
#
#
########

import math
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from icecream import ic


# Class for Neural Network
class ICM:
    def __init__(self, options, scene_const, name):
        # General Variables for plotting
        self.fig = plt.figure( figsize=(0.1*scene_const.sensor_distance*2,0.1*scene_const.goal_distance) )
        self.ax = self.fig.add_subplot(111)
        self.data_x = []
        self.data_y = []
        self.data_arrow_x = []
        self.data_arrow_y = []
        self.plot_counter = 0       # variable for numbering the plot for estimation

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

        # KERAS
        self.observation_k  = tf.keras.layers.Input( shape = ( scene_const.sensor_count, options.FRAME_COUNT), name='icm_input_sensor_frame')
        self.goal_k         = tf.keras.layers.Input( shape = ( 2, options.FRAME_COUNT), name='icm_input_goal_frame')
        self.action_k       = tf.keras.layers.Input( shape = (1,), name='icm_input_action')

        network_type = 1
        if network_type == 1:
            # Flatten
            self.h_flat1_k = tf.keras.layers.Flatten()(self.observation_k)
            self.h_flat2_k = tf.keras.layers.Flatten()(self.goal_k)
            self.h_flat3_k = tf.keras.layers.Flatten()(self.action_k)
            self.h_concat_k = tf.keras.layers.concatenate([self.h_flat1_k, self.h_flat2_k, self.h_flat3_k])

            # Dense
            self.h_dense1_k = tf.keras.layers.Dense( options.H1_SIZE, activation=options.ACT_FUNC)( self.h_concat_k )
            self.h_dense2_k = tf.keras.layers.Dense( options.H2_SIZE, activation=options.ACT_FUNC)( self.h_dense1_k )
            self.h_dense3_k = tf.keras.layers.Dense( options.H3_SIZE, activation=options.ACT_FUNC)( self.h_dense2_k )

            # Sensor Estimation
            self.out_sensor = tf.keras.layers.Dense( scene_const.sensor_count, activation='sigmoid', name='out_sensor')( self.h_dense3_k )
            # Goal Distance & Angle
            self.out_goal_dist  = tf.keras.layers.Dense( 1, activation='sigmoid', name='out_goal_dist')( self.h_dense3_k )
            self.out_goal_angle = tf.keras.layers.Dense( 1, activation='tanh', name='out_goal_angle')( self.h_dense3_k )

            self.out = tf.keras.layers.concatenate( [self.out_sensor, self.out_goal_angle, self.out_goal_dist] )

        self.model = tf.keras.models.Model( inputs = [self.observation_k, self.goal_k, self.action_k], outputs = self.out)
        keras_opt = tf.keras.optimizers.Adam(lr = options.LR)
        self.model.compile( optimizer= keras_opt,
                            loss = 'mean_squared_error' 
        )
        print('ICM MODEL')
        self.model.summary()

        # effectively computing twice to get value and maximum
        tf.keras.utils.plot_model( self.model, to_file='model_icm.png')

        return

        with tf.variable_scope(self.scope):      # Set variable scope

            ######################################:
            ## CONSTRUCTING NEURAL NETWORK
            ######################################:

            # Inputs
            # Input is [s_{t}, a_{t}] where s_t is the lidar measurements + goal and action is single number converetd to value from one hot encoding
            
            # Outputs
            # [\hat{s}_{t+1}]

            #self.observation    = tf.placeholder(tf.float32, [None, (scene_const.sensor_count+2) + 1], name='icm_input')
            self.observation    = tf.placeholder(tf.float32, [None, (scene_const.sensor_count+2)*options.FRAME_COUNT + options.ACTION_DIM], name='icm_input')
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
            # Scaled version of the next position
            #self.est_veh_xy = tf.layers.dense( inputs=self.h_s3,
                                              #units=2,
                                              #activation = None,
                                              #kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              #name="est_veh_xy"
                                         #)

            self.est_combined = tf.concat([self.est_sensor, self.est_goal_dist, self.est_goal_angle], -1)
            #self.est_combined = tf.concat([self.est_sensor, self.est_veh_xy], -1)
            ######################################:
            ## END Constructing Neural Network
            ######################################:

            # Loss Scaling Factor. \sum (w_i x_i)^2
            loss_scale = np.ones( scene_const.sensor_count+2 )*10
            loss_scale[scene_const.sensor_count] = 4000                                   # error for angle
            loss_scale[scene_const.sensor_count+1] = 100                                 # error for distance
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
    def getEstimate(self, feed, batch_size_in = 1):
        return self.model.predict(feed, batch_size=batch_size_in)
        # return self.est_combined.eval(feed_dict = feed)


    # Plot current position of the vehicle and the estimation   
    #   curr_state_sensor : dim : sensor_count x frame_count
    #   curr_state_goal   : [goal_angle;goal_distance] dim : 2 x frame_count
    #   action     : index of action
    #   veh_heading: headings of vehicle gamma in radians. 
    #                we use gamma and 0 deg -> front, +90 deg -> right, -90 deg -> left
    #   scene_const: scene constants as the class
    #   ref        : plot reference. If 'vehicle', then viewpoint from vehicle (DEFAULT). If 'ground', then view point from the ground.

    def plotEstimate(self, scene_const, options, curr_state_sensor, curr_state_goal, action, veh_heading, agent_train, ref = 'ground', save = False):
        ic.enable()
        # Change into one-hot encoded vector
        act_index = int(action)
        action = np.zeros( options.ACTION_DIM )
        action[act_index] = 1

        ####################
        # Proccess Data
        ####################
        veh_x, veh_y, arrow_x, arrow_y, goal_x, goal_y = self.getPoints(scene_const, curr_state_goal, action)
        ic( veh_x, veh_y )
        ic( goal_x, goal_y )

        return
        # Resets pos of veh to origin if viewpoint is w.r.t. the vehicle
        if ref == 'vehicle':
            veh_x = 0
            veh_y = 0

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

        if ref == 'vehicle':        
            # This works for some reason
            veh_heading = 0
        else:
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
        max_horizon     = 3
        horizon_step    = 1

        # Array storing x,y position of estimate
        state_estimate_x = []
        state_estimate_y = []

        # Radar estimation
        radar_est_x = np.zeros((max_horizon,scene_const.sensor_count))
        radar_est_y = np.zeros((max_horizon,scene_const.sensor_count))

        # Collision range from the estimated point
        radar_est_x_col = np.zeros((max_horizon,scene_const.sensor_count))
        radar_est_y_col = np.zeros((max_horizon,scene_const.sensor_count))

        # Curr state concatentated into an array
        curr_state_concat = np.reshape(np.concatenate([curr_state, action]), [-1, (scene_const.sensor_count+2) + options.ACTION_DIM])

        for i in range(0,max_horizon):
            # Get estimate and strip into an array
            temp_state = self.getEstimate( {self.observation : curr_state_concat}  )
            temp_state = np.reshape(temp_state, -1)

            # Compute the x,y coordinate
            temp_x, temp_y, _, _, _ , _ = self.getPoints( scene_const, temp_state, action )

            # resets to origin if viewpoint is w.r.t. vehicle
            if ref == 'vehicle':
                temp_x = 0
                temp_y = 0


            # Add to list
            state_estimate_x.append(temp_x)
            state_estimate_y.append(temp_y)

            # Update radar
            for k in range(0,scene_const.sensor_count):
                radar_est_x[i][k] = scene_const.sensor_distance*temp_state[k]*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_x
                radar_est_y[i][k] = scene_const.sensor_distance*temp_state[k]*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_y

                radar_est_x_col[i][k] = scene_const.collision_distance*np.sin( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_x
                radar_est_y_col[i][k] = scene_const.collision_distance*np.cos( -1*veh_heading*scene_const.angle_scale + scene_const.sensor_min_angle + k*scene_const.sensor_delta ) + temp_y

            # Get new optimal action from estimated state
            new_action = agent_train.sample_action(
                                                {
                                                    agent_train.observation : np.reshape(temp_state, (1,-1))
                                                },
                                                0,      # Get optimal action
                                                options,
                                                False
                                             )

            # Update curr state for next estimation
            new_act_index = new_action
            new_action = np.zeros(options.ACTION_DIM)
            new_action[new_act_index] = 1
            curr_state_concat = np.reshape(np.concatenate([temp_state, new_action]), [-1, (scene_const.sensor_count+2) + options.ACTION_DIM])



        ####################
        # Plot
        ####################
        
        # Goal Point
        if ref == 'vehicle':
            pass
            self.ax.scatter(goal_x,goal_y, color='blue')
        elif ref == 'ground':
            self.ax.scatter(0,scene_const.lane_len*0.5, color='blue')  # straight
            #self.ax.scatter(scene_const.turn_len*0.5 - 3,0.5*(scene_const.lane_len - scene_const.lane_width), color='blue')   # left

        # Scatter

        # Vehicle Trajectory & Input
        self.ax.scatter(self.data_x,self.data_y, color='red')

        # quiver the vehicle heading
        #self.ax.quiver(veh_x,veh_y, np.sin(veh_heading*math.pi), np.cos(veh_heading*math.pi), color='red')
        #self.ax.quiver(self.data_x,self.data_y,self.data_arrow_x,self.data_arrow_y, color='red')

        # Estimate
        self.ax.plot(state_estimate_x[0:max_horizon:horizon_step],state_estimate_y[0:max_horizon:horizon_step], color='green', label='Estimate')
        self.ax.scatter(state_estimate_x[0:max_horizon:horizon_step],state_estimate_y[0:max_horizon:horizon_step], color='green')

        # True Radar
        #   True radar points
        self.ax.scatter(radar_x,radar_y, color='red', marker = 'x')
        #   True radar lines
        self.ax.plot(radar_x,radar_y, color='red', label = 'True')
        #   True radar Collision
        self.ax.plot(radar_x_col,radar_y_col, color='red', label = 'Collision Range', linestyle = '--')

        # Radar Estimate
        color_delta = 0.9/max_horizon
        for i in range(0,max_horizon,horizon_step):
            # color, gets lighter as estimate more into the future
            radar_est_color = (0,0.5,0,1-color_delta*i) 

            # Scatter radar points
            self.ax.scatter(radar_est_x[i,:],radar_est_y[i,:], color=radar_est_color, marker = 'x')
            # Line radar points
            self.ax.plot(radar_est_x[i,:],radar_est_y[i,:], color=radar_est_color)
            # Dooted line for collision
            self.ax.plot(radar_est_x_col[i,:],radar_est_y_col[i,:], color=radar_est_color, linestyle = '--')
 
        # Figure Properties
        if ref == 'vehicle':
            self.ax.set_xlim(-scene_const.sensor_distance,scene_const.sensor_distance) 
            self.ax.set_ylim(-1,scene_const.goal_distance+5) 
        elif ref == 'ground':
            fig_width = 10
            self.ax.set_xlim(-1.2*fig_width,1.2*fig_width) 
            self.ax.set_ylim(-0.5*scene_const.lane_len,scene_const.lane_len*0.5 + 15) 
        self.ax.legend()

        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    
        # Save
        if save == True:
            self.fig.savefig('./image_dir/plot_' + str(self.plot_counter).zfill(3) + '.png')
            self.plot_counter = self.plot_counter + 1
        return

    # Compute data required for plotting from state and action values
    # Input
    #   curr_state_goal : 
    #   action     : one hot encoded vector
    #   scene_const: scene constants as the class
    #   ref        : string 
    #              'vehicle' - viewpoint from vehicle (DEFAULT)
    #              'ground'  - viewpoint from the ground
    #   goal_pos   : VEH_COUNT x 2, array of goal position [x1,y1;x2,y2;...] 
    #
    # Output
    #   veh_x, veh_y: x,y coordinate of vehicle. 
    #   arrow_x, arrow_y: x,y coordinate of arrow computed from input
    #   goal_x, goal_y : x,y coordinate of the goal point

    def getPoints(self, scene_const, curr_state_goal, action, goal_pos, veh_heading = 0):
        if curr_state_goal.shape[0] != 2:
            raise ValueError('Wrong input type')

        ic(curr_state_goal, action)

        # Break up into raw data
        # raw_sensor = curr_state[0:scene_const.sensor_count]
        raw_angle  = curr_state_goal[0,:]
        raw_dist   = curr_state_goal[1,:]

        ic(raw_dist,raw_angle)
        # Get position of vehicle from goal point distance and angle from raw data
        veh_x = -1*scene_const.goal_distance * raw_dist * np.sin( math.pi * raw_angle )        # -1 since positive distance means goal is right of vehicle. Since goal is at x=0, from goal's view, vehicle is at left of it, meaning negative.
        veh_y = scene_const.goal_distance - scene_const.goal_distance * raw_dist * np.cos( math.pi * raw_angle ) - 0.5*scene_const.lane_len     # -0.5*lane_len since veh_y think vehicle starts at origin

        # Get arrow
        # Delta of angle between each action
        action_delta = (scene_const.max_steer - scene_const.min_steer) / (5-1)

        # Calculate desired angle
        desired_angle = scene_const.max_steer - np.argmax(action) * action_delta
        arrow_x = -0.1*np.sin( math.radians( desired_angle ) )
        arrow_y = 0.1*np.cos( math.radians( desired_angle ) )

        # Get goal point assuming vehicle is the reference
        goal_y = raw_dist*scene_const.goal_distance*np.cos( raw_angle * math.pi )
        goal_x = raw_dist*scene_const.goal_distance*np.sin( raw_angle * math.pi )

        return veh_x, veh_y, arrow_x, arrow_y, goal_x, goal_y

    def getTrainableVarByName(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        trainable_vars_by_name = {var.name[len(self.scope):]: var for var in trainable_vars }   # This makes a dictionary with key being the name of varialbe without scope
        return trainable_vars_by_name
