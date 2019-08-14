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
        self.sensor_queue = []
        self.goal_queue   = []

        for i in range(0,self.options.VEH_COUNT):
            self.sensor_queue.append( deque() )
            self.goal_queue.append( deque() )

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

        # Initialize the data queues
        _, _, dDistance, gInfo = getVehicleState( self.scene_const, self.options, self.handle_dict )
        self.sensor_queue, self.goal_queue = initQueue( self.options, self.sensor_queue, self.goal_queue, dDistance, gInfo )

        return self.handle_dict, self.scene_const, direction


    # Get Observation
    # Input
    #   type - 'curr'/'next' 
    # Output
    #   structure follows the getVehicleState
    def getObservation(self, type, verbosity = 0):
        next_veh_pos, next_veh_heading, next_dDistance, next_gInfo = getVehicleState( self.scene_const, self.options, self.handle_dict)

        # Update queue
        for v in range(0,options.VEH_COUNT):
            self.sensor_queue[v].append(next_dDistance[v])
            self.sensor_queue[v].popleft()
            self.goal_queue[v].append(next_gInfo[v])
            self.goal_queue[v].popleft()

        # Get state
        # out = getVehicleState( self.scene_const, self.options, self.handle_dict )

        # Print if verbose
        if verbosity > 0:
            print('===================')
            ic(self.sensor_queue)
            ic(self.goal_queue)
            print('===================')

        return self.sensor_queue, self.goal_queue


    def updateObservation():

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
