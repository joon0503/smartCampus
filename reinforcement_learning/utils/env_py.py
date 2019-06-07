# This files contains the definition for environment class. 
# Goal is to have this function similar to the OPEN AI Gym

import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
from icecream import ic

from utils.utils_pb import (controlCamera, detectCollision, detectReachedGoal,
                            getObs, getVehicleState, initQueue, resetQueue)
from utils.utils_pb_scene_2LC import genScene, initScene, removeScene, printRewards, printSpdInfo


class env_py:
    # Initializer
    def __init__(self, options, scene_const):
       self.options = options
       self.scene_const = scene_const
       self.clientID = []
       self.handle_dict = []
       return

    # Start Simulation & Generate the Scene
    # Returns
    #   clientid - list of ids
    #   handle_dict
    def start(self): 
        print("=============================================")
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
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane100.urdf"))

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
        initScene( self.scene_const, self.options, veh_reset_list, self.handle_dict, randomize = randomize_input)               # initialize

        return self.handle_dict, self.scene_const


    # Get Observation
    # Output
    #   structure follows the getVehicleState
    def getObservation(self, verbosity = 0):
        # Get state
        out = getVehicleState( self.scene_const, self.options, self.handle_dict )

        # Print if verbose
        if verbosity > 0:
            print('===================')
            ic(out)
            print('===================')

        return out

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