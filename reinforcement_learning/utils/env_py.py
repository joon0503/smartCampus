# This files contains the definition for environment class. 
# Goal is to have this function similar to the OPEN AI Gym

import os
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
from icecream import ic

from utils.utils_pb import (controlCamera, detectCollision, detectReachedGoal,
                            getObs, getVehicleState, initQueue, resetQueue)
from utils.utils_pb_scene_2LC import (genScene, initScene_2LC, printRewards,
                                      printSpdInfo, removeScene)
from utils.genTraj_script import genTrajectory

# Add noise to the detection state
# Input
#   obs_sensor_stack : VEH_COUNT x SENSOR_COUNT*2
# Output
#   obs_sensor_stack_noismath.degrees(e : SIZE : VEH_COUNT x SENSOR_COUNT*2
#                            The detection state of the measurement data is modified with noise.
#
# Noise Adding Algorithm
#   50% of chance 
#       -do not include noise
#   50% of chance 
#       -include noise. We pick a single sensor, and scale its distance to randomly between min% to 100% of its real value. 
#        set corresponding sensor's detection to 1 (open). 
def addNoise( options, scene_const, obs_sensor_stack ):
    # Split distance and detection state
    obs_dist    = obs_sensor_stack[:,0:scene_const.sensor_count]
    obs_detect  = obs_sensor_stack[:,scene_const.sensor_count:]

    # Probability of sensoring being modified
    noise_p = options.NOISE_PROB

    # Generate random binary mask 
    sensor_mask = np.random.choice( 2, size=(options.VEH_COUNT, scene_const.sensor_count), p = [1-noise_p, noise_p]  )  

    # DO NOT modify if distance is in collision range
    sensor_mask[ obs_dist <= scene_const.collision_distance/scene_const.sensor_distance] = 0

    # Set detection state to open (1)
    obs_detect[sensor_mask != 0 ] = 1

    # Modify detected distance
    dist_mask = np.random.random( (options.VEH_COUNT,scene_const.sensor_count) )
    dist_mask[sensor_mask == 0] = 1
    obs_dist = obs_dist * dist_mask

    return np.concatenate((obs_dist,obs_detect), axis=1)


# Helper function for plotting. Compute (x,y) coordinate the of the lidar information
# This assumes, vehicle position is at 0,0
# Inputs
#   lidar
#   veh_heading
# Outputs
#   radar_x
#   radar_y 
def getLidarXY( scene_const, lidar_state, veh_heading ):
    radar_x = []
    radar_y = []

    for i in range(0,scene_const.sensor_count):
        # gamma + sensor_min_angle   is the current angle of the left most sensor
        # radar_x.append( self.scene_const.sensor_distance*curr_state[i]*np.sin( -1*veh_heading*self.scene_const.angle_scale + self.scene_const.sensor_min_angle + i*self.scene_const.sensor_delta ) + veh_x )
        # radar_y.append( self.scene_const.sensor_distance*curr_state[i]*np.cos( -1*veh_heading*self.scene_const.angle_scale + self.scene_const.sensor_min_angle + i*self.scene_const.sensor_delta ) + veh_y )

        # Angle from y-axis to sensor. CW +, CCW -
        psi_angle = veh_heading + scene_const.sensor_min_angle + i*scene_const.sensor_delta 

        # Sensor distance in meter
        sensor_dist_m = scene_const.sensor_distance * lidar_state[i]

        # Plot lidar x,y
        radar_x.append( -1 * sensor_dist_m * np.sin( -1*psi_angle ))
        radar_y.append( sensor_dist_m * np.cos( -1*psi_angle ))

    return radar_x, radar_y







class env_py:
    # Initializer
    def __init__(self, options, scene_const):
        # Basic vars 
        self.options         = options
        self.scene_const     = scene_const
        self.clientID        = []
        self.handle_dict     = []

        # Momory for data frames
        # List of deque to store data
        self.sensor_queue       = []
        self.goal_queue         = []
        self.veh_pos_queue      = []
        self.veh_heading_queue  = []

        # Add list of queue
        for i in range(0,self.options.VEH_COUNT):
            self.sensor_queue.append( deque() )
            self.goal_queue.append( deque() )
            self.veh_pos_queue.append( deque() )
            self.veh_heading_queue.append( deque() )

        # Initilialize Queue with empty data
        for i in range(0,self.options.VEH_COUNT):
            for _ in range(0,self.options.FRAME_COUNT + 1):
                self.sensor_queue[i].append( np.empty( self.scene_const.sensor_count*2 ) )
                self.goal_queue[i].append( 0 )
                self.veh_pos_queue[i].append( 0 )
                self.veh_heading_queue[i].append( 0 )

        # Memory related to rewards
        self.epi_reward_stack    = np.zeros(self.options.VEH_COUNT)                              # Holds reward of current episode
        self.epi_step_stack      = np.zeros(self.options.VEH_COUNT, dtype=int)                              # Count number of step for each vehicle in each episode

        # Goal position for each testcase (VEH_COUNT x 2) [x1,y1;x2,y2]
        self.goal_pos            = np.empty((self.options.VEH_COUNT,2), dtype=float)                              # Goal position of each vehicle

        # Plotting related variables
        self.fig = None
        self.ax  = None
        self.plot_counter = 0
        return

    # Start Simulation & Generate the Scene
    # Returns
    #   clientid - list of ids
    #   handle_dict
    def start(self): 
        print('======================================================')
        print("Starting Simulations...")
        for i in range(0,self.options.THREAD):
            if self.options.enable_GUI == True:
                # connect
                curr_ID = p.connect(p.GUI)
                self.clientID.append( curr_ID )

                # Camera
                # p.resetDebugVisualizerCamera( cameraDistance = 5, cameraYaw = 0, cameraPitch = -89, cameraTargetPosition = [0,0,0], physicsClientId = curr_ID )
                p.resetDebugVisualizerCamera( cameraDistance = 5, cameraYaw = 0, cameraPitch = -89, cameraTargetPosition = [0,0,0] )
            else:
                curr_ID = p.connect(p.DIRECT)
                self.clientID.append(curr_ID)

        # Check if all client ID is positive
        if any( id < 0 for id in self.clientID ):
            print("ERROR: Cannot establish connection to PyBullet.")
            sys.exit()

        # Simulation Setup
        for id in self.clientID:
            p.resetSimulation( id )
            p.setRealTimeSimulation( 0, id )        # Don't use real time simulation
            p.setGravity(0, 0, -9.8, id )
            p.setTimeStep( 1/60, id )

        #-----------------------
        # Scenario Generation
        #-----------------------

        # Create empty handle list & dict
        vehicle_handle      = np.zeros(self.options.VEH_COUNT, dtype=int)
        dummy_handle        = np.zeros(self.options.VEH_COUNT, dtype=int)
        case_wall_handle    = np.zeros((self.options.VEH_COUNT,self.scene_const.wall_cnt), dtype=int) 
        motor_handle        = [2, 3]
        steer_handle        = [4, 6]

        # Make handle into dict to be passed around functions
        handle_dict = {
            # 'sensor'    : sensor_handle,
            'motor'     : motor_handle,
            'steer'     : steer_handle,
            'vehicle'   : vehicle_handle,
            'dummy'     : dummy_handle,
            'wall'      : case_wall_handle,
            # 'obstacle'  : obs_handle
        }

        # Load plane
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane100.urdf"), globalScaling=10)

        # Generate Scene and get handles
        self.handle_dict, _ = genScene( self.scene_const, self.options, handle_dict, range(0,self.options.VEH_COUNT) )


        # Figure for plotting
        if self.options.DRAW == True:
            if self.options.manual == True:
                plt.ion()
                plt.show()
            self.fig = plt.figure(num=0, figsize=(4,10))
            self.ax  = self.fig.add_subplot(1,1,1)


        print("Finished starting simulations.")
        print("=============================================")
        return self.clientID, self.handle_dict

    def end(self):
        for id in self.clientID:
            p.disconnect(id)

    # Initilize scene. Remove the scene, and regenerate again
    # Input
    # course_eps : course hardness. 0 - hard, 1- easy, determine where the vehicle starts and how far is the goal point
    def initScene( self, veh_reset_list, randomize_input, course_eps = 0 ):
        # Remove
        removeScene( self.scene_const, self.options, veh_reset_list, self.handle_dict)

        # Generate the scene with updated lane_width, but not vehicles
        if randomize_input == True:
            self.scene_const.lane_width = np.random.random_sample()*(self.scene_const.MAX_LANE_WIDTH - self.scene_const.MIN_LANE_WIDTH) + self.scene_const.MIN_LANE_WIDTH

        self.handle_dict, valid_dir = genScene( self.scene_const, self.options, self.handle_dict, veh_reset_list, genVehicle = False )

        # Initilize position
        direction, goal_pos_temp = initScene_2LC( self.scene_const, self.options, veh_reset_list, self.handle_dict, valid_dir, course_eps, randomize = randomize_input)               # initialize

        # goal_pos_temp is zero if not updated, and nonzero if updated. Hence only change the goal_pos with new values
        if goal_pos_temp is not None: 
            # if none, then no update
            self.goal_pos[np.nonzero(goal_pos_temp)] = goal_pos_temp[np.nonzero(goal_pos_temp)]

        # Reset the data queues
        veh_pos_init, veh_heading_init, dDistance, gInfo = getVehicleState( self.scene_const, self.options, self.handle_dict )
        for v in range(0,self.options.VEH_COUNT):
            if v in veh_reset_list:
                for _ in range(0,self.options.FRAME_COUNT + 1):
                    # Update queue
                    self.sensor_queue[v].append(dDistance[v])
                    self.sensor_queue[v].popleft()

                    self.goal_queue[v].append(gInfo[v])
                    self.goal_queue[v].popleft()

                    self.veh_pos_queue[v].append(veh_pos_init[v])
                    self.veh_pos_queue[v].popleft()

                    self.veh_heading_queue[v].append(veh_heading_init[v])
                    self.veh_heading_queue[v].popleft()
        # self.sensor_queue, self.goal_queue, self.veh_pos_queue, self.veh_heading_queue = resetQueue( self.options, self.sensor_queue, self.goal_queue, self.veh_pos_queue, self.veh_heading_queue, veh_pos_init, veh_heading_init, dDistance, gInfo, veh_reset_list  )

        # Print Initial
        # ic('INITIAL VALUES',self.sensor_queue, self.goal_queue)

        return self.handle_dict, self.scene_const, direction


    # Get Observation
    # Input
    # old - T/F
    #       True if take the old data
    #       False take latest data
    # frame - integer from 0 ... FRAME_COUNT-1
    #       [frame 0, frame 1, frame 2, frame 3]
    #       where oldest on left and latest one right.
    #       with frame provided, it picks up a single frame, instead of block of frames
    #
    # From sensor and goal queue, get observation.
    # observation is a row vector with all frames of information concatentated to each other
    # Recall that queue stores 2*FRAME_COUNT of information. 
    # first = True: get oldest info
    # first = False: get latest info
    #
    # This work as follows. (Curr frame:3)
    #       [frame1, frame2, frame2]
    #       old = [frame1,frame2]
    #       new = [frame2,frame3]
    # Each data is (sensor_count + 2,frame_count). On each frame, last 2 elements are goal point
    # Rightmost column is the latest frame
    # Output
    #   sensor_out : [sensor_count , frame_count] 
    #   goal_out   : [2, frame_count] , first row is angle, second row is distance
    def getObservation(self, old = False, verbosity = 0, frame = None):
        # ic('getObservation:')
        # ic(np.concatenate(self.sensor_queue))
        # ic(np.concatenate(self.sensor_queue).shape)
        # ic(self.sensor_queue)
        temp_sensor = np.concatenate(self.sensor_queue).reshape(self.options.VEH_COUNT, self.options.FRAME_COUNT+1,self.scene_const.sensor_count*2)
        temp_goal   = np.concatenate(self.goal_queue).reshape(self.options.VEH_COUNT, self.options.FRAME_COUNT+1,2)
        temp_pos    = np.concatenate(self.veh_pos_queue).reshape(self.options.VEH_COUNT, self.options.FRAME_COUNT+1,2)
        temp_head   = np.concatenate(self.veh_heading_queue).reshape(self.options.VEH_COUNT, self.options.FRAME_COUNT+1,3)

        # ic(temp_sensor)
        # ic(temp_sensor.shape)
        if frame == None:
            if old == True:
                sensor_out  = temp_sensor[:,0:self.options.FRAME_COUNT,:]
                goal_out    = temp_goal[:,0:self.options.FRAME_COUNT,:]
                pos_out     = temp_pos[:,0:self.options.FRAME_COUNT,:]
                head_out    = temp_head[:,0:self.options.FRAME_COUNT,:]
            else:
                sensor_out  = temp_sensor[:,1:,:]
                goal_out    = temp_goal[:,1:,:]
                pos_out     = temp_pos[:,1:,:]
                head_out    = temp_head[:,1:,:]

            sensor_out  = np.swapaxes( sensor_out, 1, 2)
            goal_out    = np.swapaxes( goal_out, 1, 2)
            pos_out     = np.swapaxes( pos_out, 1, 2)
            head_out    = np.swapaxes( head_out, 1, 2)
        else:
            sensor_out  = temp_sensor[:,frame,:]
            goal_out    = temp_goal[:,frame,:]
            pos_out     = temp_pos[:,frame,:]
            head_out    = temp_head[:,frame,:]


        return pos_out, head_out, sensor_out, goal_out

    # Update the observation queue
    # Input
    #   type - 'curr'/'next' 
    #   add_noise = T/F. If true, add noise
    # Output
    #   None
    def updateObservation( self, reset_veh_list, add_noise = True ):
        next_veh_pos, next_veh_heading, next_dDistance, next_gInfo = getVehicleState( self.scene_const, self.options, self.handle_dict)

        if add_noise == True:
            next_dDistance = addNoise( self.options, self.scene_const, next_dDistance )

        # Update queue
        for v in reset_veh_list:
            self.sensor_queue[v].append(next_dDistance[v])
            self.sensor_queue[v].popleft()

            self.goal_queue[v].append(next_gInfo[v])
            self.goal_queue[v].popleft()

            self.veh_pos_queue[v].append(next_veh_pos[v])
            self.veh_pos_queue[v].popleft()

            self.veh_heading_queue[v].append(next_veh_heading[v])
            self.veh_heading_queue[v].popleft()
        return

    # Apply Action
    # Inputs
    # targetSteer : target angle in degrees
    def applyAction(self, targetSteer):
        for veh_index in range(self.options.VEH_COUNT):
            p.setJointMotorControlArray( self.handle_dict['vehicle'][veh_index], self.handle_dict['steer'], p.POSITION_CONTROL, targetPositions = np.repeat(targetSteer[veh_index],len(self.handle_dict['steer'])) )
            p.setJointMotorControlArray( self.handle_dict['vehicle'][veh_index], self.handle_dict['motor'], p.VELOCITY_CONTROL, targetVelocities = np.repeat(self.options.INIT_SPD,len(self.handle_dict['motor'])) )

        return


    # Step through simulation
    def step(self):            
        for q in range(0,self.options.FIX_INPUT_STEP):
            p.stepSimulation()

        if self.options.manual == True:
            input('Press Enter')

        return

    # Some info related the scenario
    def printInfo(self): 
        printRewards(self.scene_const, self.options)
        printSpdInfo(self.options)

        return

    # Given the observation, find rewards. Also returns various other information
    def getRewards(self, next_dDistance, next_veh_pos, next_gInfo, next_veh_heading ):
        reward_stack        = np.zeros( self.options.VEH_COUNT )                              # Holds rewards of current step
        epi_done            = np.zeros( self.options.VEH_COUNT) 
        epi_sucess          = np.zeros( self.options.VEH_COUNT)                              # array to keep track of whether epi succeed 
        veh_status          = np.zeros( self.options.VEH_COUNT)                              # array to keep track of whether epi succeed 

        # Handle Events
        for v in range(0,self.options.VEH_COUNT):
            # If vehicle collided, give large negative reward
            collision_detected, collision_sensor = detectCollision(next_dDistance[v], self.scene_const)
            if collision_detected == True:
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' collided! Detected Sensor : ' + str(collision_sensor) )
                print('-----------------------------------------')

                # Record data
                veh_status[v]   = self.scene_const.EVENT_COLLISION
                reward_stack[v] = self.options.FAIL_REW
                epi_done[v]     = 1

                # Resets
                # self.epi_reward_stack[v] = 0
                # self.epi_step_stack[v] = 0
                continue

            # If vehicle is at the goal point, give large positive reward
            if detectReachedGoal(next_veh_pos[v], next_gInfo[v], next_veh_heading[v], self.scene_const):
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' reached goal point')
                print('-----------------------------------------')

                veh_status[v]   = self.scene_const.EVENT_GOAL
                reward_stack[v] = self.options.GOAL_REW
                epi_done[v]     = 1
                epi_sucess[v]   = 1

                # self.epi_reward_stack[v] = 0
                # self.epi_step_stack[v] = 0
                continue

            # If over MAXSTEP
            if self.epi_step_stack[v] > self.options.MAX_TIMESTEP:
                print('-----------------------------------------')
                print('Vehicle #' + str(v) + ' over max step')
                print('-----------------------------------------')

                veh_status[v]   = self.scene_const.EVENT_OVER_MAX_STEP
                epi_done[v]     = 1

                # self.epi_reward_stack[v] = 0
                # self.epi_step_stack[v] = 0
                continue

            veh_status[v]   = self.scene_const.EVENT_FINE
            # Update rewards
            #reward_stack[v] = -(options.DIST_MUL+1/(next_dDistance[v].min()+options.MIN_LIDAR_CONST))*next_gInfo[v][1]**2 + 3*( -5 + (10/(options.ACTION_DIM-1))*np.argmin( action_stack[v] ) )
            # reward_stack[v] = -(self.options.DIST_MUL + 1/(next_dDistance[v].min()+self.options.MIN_LIDAR_CONST))*next_gInfo[v][1]**2 + 3*( -5 + (10/(self.options.ACTION_DIM-1))*np.argmin( action_stack[v] ) )
            # reward_stack[v] = -(self.options.DIST_MUL + 1/(next_dDistance[v].min()+self.options.MIN_LIDAR_CONST))*next_gInfo[v][1]**2
            reward_stack[v] = -(self.options.DIST_MUL)*next_gInfo[v][1]**2

        # Update cumulative rewards
        self.epi_step_stack = self.epi_step_stack + 1
        for v in range(0,self.options.VEH_COUNT):
            self.epi_reward_stack[v] = self.epi_reward_stack[v] + reward_stack[v]*(self.options.GAMMA**self.epi_step_stack[v])

        if self.options.VERBOSE == True:
            ic(reward_stack)
        return reward_stack, veh_status, epi_done, epi_sucess


    # Reset the epi_step_stack and epi_reward_stack
    def resetRewards(self, veh_status):
        for v, veh_status_val in enumerate(veh_status):
            if veh_status_val != self.scene_const.EVENT_FINE:
                self.epi_reward_stack[v] = 0
                self.epi_step_stack[v] = 0

        return

    # Plot visualization
    # Given the handle to figures, it plots the position, heading and LIDAR information w.r.t. ground 
    # Inputs
    #   veh_idx : index of the vehicle
    #   frame   : number of frames to plot. 0 means options.FRAME_COUNT
    #   save    : T/F, True means save figure
    #   predict : integer, number of prediction into the future. 0 means dont plot prediction
    # Outputs
    #   None
    def plotVehicle(self, veh_idx = 0, frame = 0, save = False, predict = 0, network_model = None):
        if frame == 0:
            frame = self.options.FRAME_COUNT

        if predict < 0:
            ic(predict)
            raise ValueError('predict cannot be negative')
        else:
            if network_model == None:
                raise ValueError('Must provide network to use prediction')

        # Clear figure
        self.ax.clear()
        self.ax.set_xlim([-0.5*self.scene_const.turn_len,0.5*self.scene_const.turn_len])
        self.ax.set_ylim([0,self.scene_const.lane_len*0.7])

        #------------------------
        # Plot Goal Position
        #------------------------
        self.ax.scatter(self.goal_pos[veh_idx][0], self.goal_pos[veh_idx][1], color='blue')

        #------------------------
        # Plot Vehicle. Stronger color means more recent position
        #------------------------

        # Color parameters
        color_start = 0.2
        color_end   = 1.0
        color_delta = (color_end - color_start)/frame

        # Plot position and heading of last few frames
        for i in range(0,frame):
            # Saturate color
            veh_color = (1,0,0,color_start + i*color_delta)

            # Plot Position
            # Add 1 to counter since we are saving 1 more data points (#0, #1, #2, #3, #4) when Frame is 5, with #4 being the latest data
            self.ax.scatter( self.veh_pos_queue[veh_idx][i+1][0], self.veh_pos_queue[veh_idx][i+1][1], color=veh_color)

            # Plot Heading
            # veh_heading = self.veh_heading_queue[veh_idx][i+1][2]
            # self.ax.quiver( self.veh_pos_queue[veh_idx][i+1][0], self.veh_pos_queue[veh_idx][i+1][1], np.sin(veh_heading), np.cos(veh_heading), color = veh_color)

        #------------------------
        # Plot LIDAR Information 
        #------------------------
        radar_x = []
        radar_y = []
        veh_x   = self.veh_pos_queue[veh_idx][-1][0]
        veh_y   = self.veh_pos_queue[veh_idx][-1][1]
        veh_heading = self.veh_heading_queue[veh_idx][-1][2]
        curr_state = self.sensor_queue[veh_idx][-1][0:self.scene_const.sensor_count]
        curr_detect = self.sensor_queue[veh_idx][-1][self.scene_const.sensor_count:]

        radar_x, radar_y = getLidarXY( self.scene_const, curr_state, veh_heading)

        # Place marker at LIDAR x,y
        for i in range(0,self.scene_const.sensor_count):
            if curr_detect[i] == 1:
                marker_color = 'green'
            else:
                marker_color = 'red'

            self.ax.scatter(radar_x[i] + veh_x,radar_y[i] + veh_y, marker='x', color=marker_color)

        # Lines connecting detected points to each other
        # self.ax.plot(radar_x,radar_y, color='red')

        # Lines connecting point to vehicle
        for i in range(0,self.scene_const.sensor_count):
            if curr_detect[i] == 1:
                line_color = (0,1,0,0.2)
            else:
                line_color = (1,0,0,0.2)
            self.ax.plot([veh_x, veh_x + radar_x[i]],[veh_y, veh_y + radar_y[i]], color=line_color )

        #----------------------------------
        # Plot Prediction
        #----------------------------------
        if predict > 0:
            # Obtain Prediction

            # put data into right format. Only get latest frames
            predict_state = np.array(self.sensor_queue)[veh_idx].T
            predict_state = np.expand_dims(predict_state[:,1:], 0)

            predict_goal  = np.array(self.goal_queue)[veh_idx].T
            predict_goal  = np.expand_dims(predict_goal[:,1:],0)

            traj_est, lidar_est = genTrajectory(
                        self.options,           # option
                        self.scene_const,       # scene_const
                        np.array([[0,0]]),      # veh position, single vehicle, at origin
                        np.expand_dims(veh_heading,0), 
                        predict_state, 
                        predict_goal, 
                        network_model, 
                        predict
                    )

            predict_veh_x = traj_est[0,0,:]
            predict_veh_y = traj_est[0,1,:]

            # Color parameters
            color_start = 1.0
            color_end   = 0.2
            color_delta = (color_start - color_end)/predict

            # Plot position and heading of last few frames
            for i in range(0,predict):
                # Saturate color
                veh_color = (0,0,1,color_start - i*color_delta)

                # Plot Position
                self.ax.scatter( veh_x + predict_veh_x[i], veh_y + predict_veh_y[i], color=veh_color)

                # Plot Heading
                # veh_heading = self.veh_heading_queue[veh_idx][i+1][2]
                # self.ax.quiver( self.veh_pos_queue[veh_idx][i+1][0], self.veh_pos_queue[veh_idx][i+1][1], np.sin(veh_heading), np.cos(veh_heading), color = veh_color)
                # Plot prediction

            # Plot Predicted LIDAR Information
            # lidar_est : VEH_COUNT x SENSOR_COUNT*2 X PREDICTION_STEP
            # FIXME: lidar_est seems wrong
            ic(lidar_est)

            # For each prediction
            for i in range(0,predict):
                # Get LIDAR x,y. 
                # FIXME: Need to get veh_heading from estimation
                radar_x, radar_y = getLidarXY(self.scene_const, lidar_est[veh_idx][0:self.scene_const.sensor_count], 0 )

                for j in range(0,self.scene_const.sensor_count):
                    if lidar_est[veh_idx][self.scene_const.sensor_count + j][i] == 1:
                        marker_color = 'green'
                    else:
                        marker_color = 'red'

                    # FIXME: Disabled plotting lidar for now
                    # self.ax.scatter(radar_x[j] + veh_x + predict_veh_x[i], radar_y[j] + veh_y + predict_veh_y[i], marker='x', color=marker_color)


        #----------------------------------
        # Save the figure if enabled
        #----------------------------------
        if save == True:
            self.fig.savefig('./image_dir/plot_' + str(self.plot_counter).zfill(3) + '.png')
            self.plot_counter = self.plot_counter + 1

        return


    # # Get estimated position and heading of vehicle
    # def getVehicleEstimation(self, oldPosition, oldHeading, targetSteer_k):
    #     vLength = 2.1
    #     vel     = self.options.INIT_SPD
    #     delT    = self.options.CTR_FREQ*self.options.FIX_INPUT_STEP

    #     newPosition = np.zeros([self.options.VEH_COUNT,2])
    #     newHeading  = np.zeros([self.options.VEH_COUNT])

    #     newPosition[:,0] = oldPosition[:,0] + vel * np.cos(oldHeading)*delT
    #     newPosition[:,1] = oldPosition[:,1] + vel * np.sin(oldHeading)*delT
    #     newHeading       = oldHeading + vel/vLength*np.tan(targetSteer_k)*delT

    #     # for v in range(0,self.options.VEH_COUNT):
    #         # newPosition[v,0]=oldPosition[v,0]+vel*np.cos(oldHeading[v])*delT
    #         # newPosition[v,1]=oldPosition[v,1]+vel*np.sin(oldHeading[v])*delT
    #         # newHeading[v]=oldHeading[v]+vel/vLength*math.tan(targetSteer_k[v])*delT

    #     return newPosition, newHeading

    # # Get estimated goal length and heading
    # def getGoalEstimation(self, veh_pos, veh_heading):
    #     gInfo           = np.zeros([self.options.VEH_COUNT,2])
    #     goal_handle     = self.handle_dict['dummy']

    #     for k in range(0,self.options.VEH_COUNT):
    #         # To compute gInfo, get goal position
    #         g_pos, _ = p.getBasePositionAndOrientation( goal_handle[k] )

    #         # Calculate the distance
    #         # ic(g_pos[0:2],veh_pos[k])
    #         delta_distance  = np.array(g_pos[0:2]) - np.array(veh_pos[k])  # delta x, delta y
    #         gInfo[k][1]     = np.linalg.norm(delta_distance) / self.scene_const.goal_distance

    #         # calculate angle. 90deg + angle (assuming heading north) - veh_heading
    #         gInfo[k][0] = math.atan(abs(delta_distance[0]) / abs(delta_distance[1]))  # delta x / delta y, and scale by pi 1(left) to -1 (right)
    #         if delta_distance[0] < 0:
    #             # Goal is left of the vehicle, then -1
    #             gInfo[k][0] = gInfo[k][0] * -1

    #         # Scale with heading
    #         gInfo[k][0] = -1 * (math.pi * 0.5 - gInfo[k][0] - veh_heading[k]) / (math.pi / 2)

    #     return gInfo

    # # Predict next LIDAR state
    # def predictLidar(self, oldPosition, oldHeading, oldSensor, newPosition, newHeading):
    #     oldSensorState  = oldSensor[:,self.scene_const.sensor_count:]
    #     # oldSensor=oldSensor[:,0:self.scene_const.sensor_count-1]
    #     oldSensor       = oldSensor[:,0:self.scene_const.sensor_count]

    #     seqAngle        = np.array([(self.scene_const.sensor_count-1-i)*np.pi/(self.scene_const.sensor_count-1) for i in range(self.scene_const.sensor_count)])
    #     newSensor       = np.zeros((self.options.VEH_COUNT,self.scene_const.sensor_count))
    #     newSensorState  = np.ones((self.options.VEH_COUNT,self.scene_const.sensor_count)) # 0:closed & 1:open

    #     for v in range(0,self.options.VEH_COUNT):
    #         oldC,oldS   = np.cos(oldHeading[v]-np.pi/2),np.sin(oldHeading[v]-np.pi/2)
    #         oldRot      = np.array([[oldC,oldS],[-oldS,oldC]])
    #         newC,newS   = np.cos(newHeading[v]-np.pi/2),np.sin(newHeading[v]-np.pi/2)
    #         newRot      = np.array([[newC,newS],[-newS,newC]])

    #         # Transformation oldSensor w.r.t vehicle's next position and heading.
    #         temp1   = self.scene_const.sensor_distance*np.transpose([oldSensor[v],oldSensor[v]])
    #         temp2   = np.transpose([np.cos(seqAngle), np.sin(seqAngle)])
    #         oldLidarXY = np.multiply(temp1,temp2)
    #         oldLidarXY = np.tile(oldPosition[v, :], [self.scene_const.sensor_count, 1]) \
    #                     + np.matmul(oldLidarXY, np.transpose(oldRot))
    #         oldTransXY = oldLidarXY-np.tile(newPosition[v,:], [self.scene_const.sensor_count,1])
    #         oldTransXY = np.matmul(oldTransXY,np.transpose(newRot))

    #         # Compute newSensor w.r.t vehicle's next position and heading
    #         newTransXY = self.scene_const.sensor_distance*np.transpose([np.cos(seqAngle), np.sin(seqAngle)])

    #         # Remove out-range points, i.e. y<0
    #         reducedLidarXY=np.array([])
    #         for i in range(0,self.scene_const.sensor_count):
    #             if oldTransXY[i,1]>=0:
    #                 reducedLidarXY = np.append(reducedLidarXY, oldTransXY[i,:])
    #         length=reducedLidarXY.shape[0]
    #         reducedLidarXY=np.reshape(reducedLidarXY, (-1,2))

    #         # Find intersection between line1(newPosition,newTransXY) & line2(two among reducedLidarXY)
    #         newLidarXY=np.zeros([self.scene_const.sensor_count,2])
    #         flag=np.ones([self.scene_const.sensor_count])
    #         length=reducedLidarXY.shape[0]
    #         for i in range(0,self.scene_const.sensor_count):
    #             for j in range(0,length-1):
    #                 l=j+1
    #                 # If intersection is found, then pass
    #                 if flag[i]==0:
    #                     pass

    #                 A=np.array([[newTransXY[i][1],-newTransXY[i][0]],[reducedLidarXY[l][1]-reducedLidarXY[j][1],-reducedLidarXY[l][0]+reducedLidarXY[j][0]]])
    #                 det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    #                 if det==0:
    #                     pass
    #                 b=np.array([0,np.dot(A[1,:],reducedLidarXY[j,:])])
    #                 try:
    #                     x=np.linalg.solve(A,b)
    #                 except:
    #                     pass

    #                 # check the point between two pair points
    #                 # if np.dot(A[1,:],newTransXY[i]-x)*np.dot(A[1,:],-x)<=0 \
    #                 #         and np.dot(A[0,:],reducedLidarXY[j]-x)*np.dot(A[0,:],reducedLidarXY[l]-x)<=0:
    #                 if np.dot(A[0, :], reducedLidarXY[j] - x) * np.dot(A[0, :], reducedLidarXY[l] - x) <= 0:
    #                     newLidarXY[i,:]=x
    #                     newSensorState[v,i]=oldSensorState[v,j]*oldSensorState[v,l] # closed if there exist at least closed point
    #                     flag[i]=0

    #         # Compute newSensorOut
    #         newLidarXY=newLidarXY+1.5*self.scene_const.collision_distance*np.transpose([np.cos(seqAngle),np.sin(seqAngle)])*np.transpose([flag,flag])
    #         newSensor[v,:]=np.linalg.norm(newLidarXY,ord=2,axis=1)/self.scene_const.sensor_distance

    #     return newSensor, newSensorState