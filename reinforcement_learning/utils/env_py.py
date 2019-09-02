# This files contains the definition for environment class. 
# Goal is to have this function similar to the OPEN AI Gym

import os
import sys
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
from icecream import ic

from utils.utils_pb import (controlCamera, detectCollision, detectReachedGoal,
                            getObs, getVehicleState, initQueue, resetQueue)
from utils.utils_pb_scene_2LC import (genScene, initScene_2LC, printRewards,
                                      printSpdInfo, removeScene)


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
                self.sensor_queue[i].append( np.empty( self.scene_const.sensor_count) )
                self.goal_queue[i].append( 0 )
                self.veh_pos_queue[i].append( 0 )
                self.veh_heading_queue[i].append( 0 )

        # Memory related to rewards
        self.epi_reward_stack    = np.zeros(self.options.VEH_COUNT)                              # Holds reward of current episode
        self.epi_step_stack      = np.zeros(self.options.VEH_COUNT, dtype=int)                              # Count number of step for each vehicle in each episode

        # Goal position for each testcase (VEH_COUNT x 2) [x1,y1;x2,y2]
        self.goal_pos            = np.empty((self.options.VEH_COUNT,2), dtype=float)                              # Count number of step for each vehicle in each episode

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
        self.handle_dict = genScene( self.scene_const, self.options, handle_dict, range(0,self.options.VEH_COUNT) )

        print("Finished starting simulations.")
        print("=============================================")
        return self.clientID, self.handle_dict

    def end(self):
        for id in self.clientID:
            p.disconnect(id)

    # Initilize scene. Remove the scene, and regenerate again
    def initScene( self, veh_reset_list, randomize_input ):
        # Remove
        removeScene( self.scene_const, self.options, veh_reset_list, self.handle_dict)

        # Generate the scene with updated lane_width, but not vehicles
        if randomize_input == True:
            self.scene_const.lane_width = np.random.random_sample()*4.5 + 3.5
        self.handle_dict = genScene( self.scene_const, self.options, self.handle_dict, veh_reset_list, genVehicle = False )

        # Initilize position
        direction, goal_pos_temp = initScene_2LC( self.scene_const, self.options, veh_reset_list, self.handle_dict, randomize = randomize_input)               # initialize

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
        temp_sensor = np.concatenate(self.sensor_queue).reshape(self.options.VEH_COUNT, self.options.FRAME_COUNT+1,self.scene_const.sensor_count)
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
            #     goal_out   = np.empty((self.options.VEH_COUNT, 2, self.options.FRAME_COUNT))
            #     pos_out    = np.empty((self.options.VEH_COUNT, 2, self.options.FRAME_COUNT))
            #     head_out   = np.empty((self.options.VEH_COUNT, 3, self.options.FRAME_COUNT))

            # for v in range(0,self.options.VEH_COUNT):
            #     if old == True:
            #         # sensor_stack    = np.concatenate(self.sensor_queue)[0:self.scene_const.sensor_count * self.options.FRAME_COUNT]
            #         # goal_stack      = np.concatenate(self.goal_queue)[0:2*self.options.FRAME_COUNT]
            #         sensor_out[v]    = np.concatenate(self.sensor_queue)[0:self.options.FRAME_COUNT-1,:].transpose()
            #         goal_out[v]      = np.concatenate(self.goal_queue)[0:self.options.FRAME_COUNT-1,:].transpose()
            #         pos_out[v]       = np.concatenate(self.veh_pos_queue)[0:self.options.FRAME_COUNT-1,:].transpose()
            #         head_out[v]      = np.concatenate(self.veh_heading_queue)[0:self.options.FRAME_COUNT-1,:].transpose()
            #     else:
            #         sensor_out[v]    = np.concatenate(self.sensor_queue)[1:,:].transpose()
            #         goal_out[v]      = np.concatenate(self.goal_queue)[1:,:].transpose()
            #         pos_out[v]       = np.concatenate(self.veh_pos_queue)[1:,:].transpose()
            #         head_out[v]      = np.concatenate(self.veh_heading_queue)[1:,:].transpose()
            #         # sensor_stack    = np.concatenate(self.sensor_queue)[self.scene_const.sensor_count:]
            #         # goal_stack      = np.concatenate(self.goal_queue)[2:]
        else:
            sensor_out  = temp_sensor[:,frame,:]
            goal_out    = temp_goal[:,frame,:]
            pos_out     = temp_pos[:,frame,:]
            head_out    = temp_head[:,frame,:]
            # sensor_out = np.empty((self.options.VEH_COUNT, self.scene_const.sensor_count, 1))
            # goal_out   = np.empty((self.options.VEH_COUNT, 2, 1))
            # pos_out    = np.empty((self.options.VEH_COUNT, 2, 1))
            # head_out   = np.empty((self.options.VEH_COUNT, 3, 1))

            # for v in range(0,self.options.VEH_COUNT):
            #     sensor_out[v]    = np.concatenate(self.sensor_queue)[frame,:].transpose()
            #     goal_out[v]      = np.concatenate(self.goal_queue)[frame,:].transpose()
            #     pos_out[v]       = np.concatenate(self.veh_pos_queue)[frame,:].transpose()
            #     head_out[v]      = np.concatenate(self.veh_heading_queue)[frame,:].transpose()

        # out = np.transpose(np.hstack((sensor_stack.reshape(self.options.FRAME_COUNT,-1), goal_stack.reshape(self.options.FRAME_COUNT,-1))))
        # return np.transpose(sensor_stack.reshape(self.options.FRAME_COUNT,-1)), np.transpose(goal_stack.reshape(self.options.FRAME_COUNT,-1))
        return pos_out, head_out, sensor_out, goal_out

    # Update the observation queue
    # Input
    #   type - 'curr'/'next' 
    # Output
    #   None
    def updateObservation( self, reset_veh_list ):
        next_veh_pos, next_veh_heading, next_dDistance, next_gInfo = getVehicleState( self.scene_const, self.options, self.handle_dict)

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
            # reward_stack[v] = -(options.DIST_MUL + 1/(next_dDistance[v].min()+options.MIN_LIDAR_CONST))*next_gInfo[v][1]**2 + 3*( -5 + (10/(options.ACTION_DIM-1))*np.argmin( action_stack[v] ) )
            reward_stack[v] = -(self.options.DIST_MUL)*next_gInfo[v][1]**2

        # Update cumulative rewards
        self.epi_step_stack = self.epi_step_stack + 1
        for v in range(0,self.options.VEH_COUNT):
            self.epi_reward_stack[v] = self.epi_reward_stack[v] + reward_stack[v]*(self.options.GAMMA**self.epi_step_stack[v])

        return reward_stack, veh_status, epi_done, epi_sucess


    # Reset the epi_step_stack and epi_reward_stack
    def resetRewards(self, veh_status):
        for v, veh_status_val in enumerate(veh_status):
            if veh_status_val != self.scene_const.EVENT_FINE:
                self.epi_reward_stack[v] = 0
                self.epi_step_stack[v] = 0

        return
