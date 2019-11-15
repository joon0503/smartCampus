import math
import os
import random

import numpy as np
import pybullet as p
import pybullet_data
from icecream import ic

#########################################
# Files for generating the scene
#########################################
# Create Wall
# Input
#   size : [x,y,z]. Note: Size is halved during the creation. So if you want to create box with 1m, then input should be 2
#   pos  : [x,y,z]


def createWall(size, pos):
    temp = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=size),
                             baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=[1, 1, 1, 1], halfExtents=size), basePosition=pos)
    if temp < 0:
        raise ValueError("Failed to create multibody at createWall")
    else:
        return temp

# Create Goal Point
# Input
#  size : radius, single number
# Output
#  goal_point id


def createGoal(size, pos):
    temp = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=size),
                             baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=size), basePosition=pos)
    if temp < 0:
        raise ValueError("Failed to create multibody at createGoal")
    else:
        return temp

# Create T-Intersection
# Input
#   x_pos : x position of the test case
#   scene_const
#   openWall : T/F
# Note:
# randomization is doen in initScene
# Output
#   goal_id : id of the goal point
#   wall_handle_list : array of wall's handle
#   valid_dir : 0/1/2 : left,middle,right

def createTee(x_pos, y_pos, scene_const, openWall=True):
    wall_handle_list = []

    # If openWall = True, then randomly choose a valid direction and close walls for other paths
    if openWall == False:
        valid_dir = np.random.random(1)
        if 0 <= valid_dir and valid_dir <= (1/3):
            # Goal on left
            left_len  = 0
            right_len = scene_const.lane_width*2
            valid_dir = 0
        elif (1/3) <= valid_dir and valid_dir <= (2/3):
            # Goal on right
            left_len  = scene_const.lane_width*2
            right_len = 0
            valid_dir = 2
        elif (2/3) <= valid_dir and valid_dir <= 1:
            # Goal on straight
            left_len  = scene_const.lane_width*2
            right_len = scene_const.lane_width*2
            valid_dir = 1
        else:
            ic(valid_dir)
            raise ValueError('Invalid direction')

    # Walls left & right
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_len + left_len, scene_const.wall_h], [
                            x_pos - scene_const.lane_width*0.5, y_pos, 0]))      # left
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_len + right_len, scene_const.wall_h], [
                            x_pos + scene_const.lane_width*0.5, y_pos, 0]))      # right

    # Walls front & back
    wall_handle_list.append(createWall([0.5*scene_const.turn_len, 0.02, scene_const.wall_h], [
                            x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width, 0]))               # front
    wall_handle_list.append(createWall([0.5*scene_const.lane_width, 0.02, scene_const.wall_h], [
                            x_pos, y_pos - 1*scene_const.lane_len*0.5, 0]))              # back

    # Walls at intersection
    wall_len = 0.5*(scene_const.turn_len - scene_const.lane_width)
    wall_handle_list.append(createWall([0.5*wall_len, 0.02, scene_const.wall_h], [
                            x_pos + 0.5*scene_const.lane_width + 0.5*wall_len, y_pos + scene_const.lane_len*0.5, 0]))
    wall_handle_list.append(createWall([0.5*wall_len, 0.02, scene_const.wall_h], [
                            x_pos - 0.5*scene_const.lane_width - 0.5*wall_len, y_pos + scene_const.lane_len*0.5, 0]))

    # Walls at the end
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_width, scene_const.wall_h], [
                            x_pos - 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0]))
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_width, scene_const.wall_h], [
                            x_pos + 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0]))

    # Create Obstacle
    obs_width = scene_const.obs_w * scene_const.lane_width
    wall_handle_list.append(createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_pos + 0.5*(
        1-scene_const.obs_w)*scene_const.lane_width, scene_const.lane_len*0.3, 0]))      # right

    # Create Goal point
    goal_z = 1.0
    goal_y = y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5
    goal_x      = x_pos
    goal_id = createGoal( 0.1, [ goal_x, goal_y, goal_z])



    # if direction == 0:
    #     goal_id = createGoal(0.1, [x_pos - scene_const.turn_len*0.5 + 1.0,
    #                                y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z])
    # elif direction == 1:
    # elif direction == 2:
    #     goal_id = createGoal(0.1, [x_pos + scene_const.turn_len*0.5 - 1.0,
    #                                y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z])
    # else:
    #     raise ValueError('Undefined direction')

    return goal_id, wall_handle_list, valid_dir

# Generate the scene
# Input
#   genVehicle     : T/F. If true, generate vehicle, if not don't generate vehicle
#   reset_case_list: list of testcase numbers which we generate the scene
# Output
#   handle_dict
#   valid_dir : 1 x VEH_COUNT, each index 0/1/2 = left/middle/right

def genScene(scene_const, options, handle_dict, reset_case_list, genVehicle=True):
    if len(reset_case_list) == 0:
        return handle_dict, None

    if options.VEH_COUNT % options.X_COUNT != 0:
        raise ValueError('VEH_COUNT=' + str(options.VEH_COUNT) + ' must be multiple of options.X_COUNT=' + str(options.X_COUNT))

    valid_dir = np.empty(options.VEH_COUNT)
    # Generate test cases & vehicles
    for j in range(int(options.VEH_COUNT/options.X_COUNT)):
        for i in range(options.X_COUNT):
            # print("Creating x:", i, " y:", j)

            # Unrolled index
            u_index = options.X_COUNT*j + i

            if u_index in reset_case_list:
                # Generate T-intersection for now
                handle_dict['dummy'][u_index], handle_dict['wall'][u_index], valid_dir[u_index] = createTee( i*scene_const.case_x, j*scene_const.case_y, scene_const, openWall = False )

                # Save vehicle handle
                if genVehicle == True:
                    temp = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"), basePosition=[ i*scene_const.case_x, j*scene_const.case_y + scene_const.veh_init_y, 0], globalScaling=scene_const.veh_scale, useFixedBase=False, baseOrientation=[0, 0, 0.707, 0.707])
                    handle_dict['vehicle'][u_index] = temp

                # Resets all wheels
                for wheel in range(p.getNumJoints(handle_dict['vehicle'][u_index])):
                    # print("joint[",wheel,"]=", p.getJointInfo(vehicle_handle[u_index],wheel))
                    p.setJointMotorControl2(handle_dict['vehicle'][u_index], wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                    p.getJointInfo(handle_dict['vehicle'][u_index], wheel)

    return handle_dict, valid_dir

# Remove the scene


def removeScene(scene_const, options, veh_reset_list, handle_dict):
    for veh_index in veh_reset_list:
        # remove goal point
        p.removeBody(handle_dict['dummy'][veh_index])
        handle_dict['dummy'][veh_index] = -1

        # remove walls
        for index, entry in enumerate(handle_dict['wall'][veh_index]):
            p.removeBody( entry )
            handle_dict['wall'][veh_index][index] = -1

    return handle_dict

# Initialize test cases. Reset position of vehicle and obstacle
# Input
# veh_index_list : vehicle reset list
# course_eps       : parameter for curriculum learning. [0,1]. 1 means easy, 0 means hard
# valid_dir      : 1 x VEH_COUNT, each index 0/1/2 = left/middle/right
# Output
#   direction : info about randomized testcase. 0/1/2 - left/straight/right
#   goal_pos  : VEH_COUNT x 2 array of goal position
def initScene_2LC(scene_const, options, veh_index_list, handle_dict, valid_dir, course_eps = 0, randomize=False):
    # If reset list is empty, just return
    if len(veh_index_list) == 0:
        return None, None 

    vehicle_handle      = handle_dict['vehicle']
    steer_handle        = handle_dict['steer']
    motor_handle        = handle_dict['motor']
    case_wall_handle    = handle_dict['wall']

    # Output
    direction           = -1*np.ones(options.VEH_COUNT)           # 0/1/2: left/straight/right
    goal_pos            = np.zeros((options.VEH_COUNT,2))

    # obs_handle     = handle_dict['obstacle']

    for veh_index in veh_index_list:
        # Reset position of vehicle. Randomize x-position if enabled
        x_index = veh_index % options.X_COUNT
        y_index = int(veh_index / options.X_COUNT)

        if randomize == False:
            p.resetBasePositionAndOrientation(vehicle_handle[veh_index], [ scene_const.case_x * x_index, scene_const.case_y * y_index + scene_const.veh_init_y, 0], [0, 0, 0.707, 0.707])
        else:
            # randomize x_position between -0.5*lane_width + 0.6 ~ 0.5*lane_width - 0.6
            # x_pos = (random.uniform()-0.5)-1*scene_const.lane_width * 0.5 + 0.6, scene_const.lane_width*0.5 - 0.6)
            x_pos = scene_const.MIN_X_POS + np.random.random(1)*(scene_const.MAX_X_POS - scene_const.MIN_X_POS)

            # randomize y_position
            y_pos = np.random.random(1) * scene_const.lane_len * 0.5

            # If y_pos lies between obstacle, just start closer to the goal point 
            # TODO : Randomize the starting point even in this case?
            if y_pos >= scene_const.MIN_OBS_Y_POS and y_pos <= scene_const.MAX_OBS_Y_POS:
                y_pos = scene_const.MAX_OBS_Y_POS

            # Reset position and heading
            p.resetBasePositionAndOrientation(
                vehicle_handle[veh_index], 
                [ 
                    scene_const.case_x * x_index + x_pos, 
                    scene_const.case_y * y_index + scene_const.veh_init_y + y_pos, 
                    0
                ], 
                [
                    0, 
                    0, 
                    0.707, 
                    0.707
                ]
            )

            # Reset speed
            # FIXME: Need to find conversion between INIT_SPD to y-axis velocity. Currently if INIT_SPD is used, it's way too fast
            p.resetBaseVelocity(vehicle_handle[veh_index],[0,2,0])

        # Reset steering & motor speed. TODO: Perhaps do this with setMotorArray to do it in a single line?
        for s in steer_handle:
            p.resetJointState(vehicle_handle[veh_index], s, targetValue=0)

        for m in motor_handle:
            p.resetJointState(
                vehicle_handle[veh_index], m, targetValue=0, targetVelocity=options.INIT_SPD)

        # Randomize the obstacle
        #   y position of obstable at 0.3*lane_len
        if randomize == True:
            obs_width = scene_const.obs_w * scene_const.lane_width
            temp_prob = random.uniform(0, 1)
            p.removeBody(case_wall_handle[veh_index][8])
            if temp_prob > (2/3):
                # right
                direction[veh_index] = 2
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x + 0.5*( 1-scene_const.obs_w)*scene_const.lane_width, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])       # right
            elif temp_prob > (1/3):
                # middle
                direction[veh_index] = 1
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])       # right
            else:
                # left
                direction[veh_index] = 0
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x - 0.5*( 1-scene_const.obs_w)*scene_const.lane_width, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])      # right


        # Randomize the goal point                
        if randomize == True:
            # Define varialbes 
            dummy_handle    = handle_dict['dummy']
            goal_z          = 1.0       

            # Randomize x_position of goal. Based on curriculum 
            if valid_dir[veh_index] == 0:
                x_pos           = random.uniform(-1*scene_const.turn_len*0.5 + 1.0, 0) * (1 - course_eps)
            elif valid_dir[veh_index] == 1:
                x_pos           = random.uniform(-1.0, 1.0) * (1 - course_eps)
            elif valid_dir[veh_index] == 2:
                x_pos           = random.uniform(0, scene_const.turn_len*0.5 - 1.0) * (1 - course_eps)
            else:
                ic(valid_dir, veh_index)
                raise ValueError('Invalid direction at valid_dir[veh_index]')


            # Remove existing goal point
            p.removeBody( dummy_handle[veh_index] )

            # Create new goal point
            dummy_handle[veh_index] = createGoal(0.1, 
                [
                    x_pos + x_index * scene_const.case_x, 
                    y_index * scene_const.case_y + 0.5*scene_const.lane_len + 0.5*scene_const.lane_width, 
                    goal_z
                ]
            )

            goal_pos[veh_index,0] = x_pos + x_index * scene_const.case_x
            goal_pos[veh_index,1] = y_index * scene_const.case_y + 0.5*scene_const.lane_len + 0.5*scene_const.lane_width

    return direction, goal_pos

# Calculate Approximate Rewards for variaous cases
def printRewards( scene_const, options ):
    # Some parameters
    veh_speed = options.INIT_SPD*0.1  # m/s. This is how current test case works

    # Time Steps
    #dt_code = scene_const.dt * options.FIX_INPUT_STEP

    # Expected Total Time Steps
    control_freq = 0.1  # 0.1 seconds per step
    total_step = (scene_const.goal_distance/veh_speed) * (1/control_freq)

    # Reward at the end
    rew_end = 0
    for i in range(0, int(total_step)):
        goal_distance = scene_const.goal_distance - i*control_freq*veh_speed 
        if i != total_step-1:
            rew_end = rew_end -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_end = rew_end + options.GOAL_REW*(options.GAMMA**i) 

    # Reward at Obs
    rew_obs = 0
    for i in range(0, int(total_step*0.5)):
        goal_distance = scene_const.goal_distance - i*control_freq*veh_speed 
        if i != int(total_step*0.5)-1:
            rew_obs = rew_obs -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_obs = rew_obs + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 75%
    rew_75 = 0
    for i in range(0, int(total_step*0.75)):
        goal_distance = scene_const.goal_distance - i*control_freq*veh_speed 
        if i != int(total_step*0.75)-1:
            rew_75 = rew_75 -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_75 = rew_75 + options.FAIL_REW*(options.GAMMA**i) 

    # Reward at 25%
    rew_25 = 0
    for i in range(0, int(total_step*0.25)):
        goal_distance = scene_const.goal_distance - i*control_freq*veh_speed 
        if i != int(total_step*0.25)-1:
            rew_25 = rew_25 -options.DIST_MUL*(goal_distance/scene_const.goal_distance)**2*(options.GAMMA**i) 
        else:
            rew_25 = rew_25 + options.FAIL_REW*(options.GAMMA**i) 

    # EPS Info
    

    ########
    # Print Info
    ########

    print("======================================")
    print("======================================")
    print("        REWARD ESTIMATION")
    print("======================================")
    # print("Control Frequency (s)  : ", options.CTR_FREQ)
    print("Control Frequency (s)  : ", control_freq)
    print("Expected Total Step    : ", total_step)
    print("Expected Reward (25)   : ", rew_25)
    print("Expected Reward (Obs)  : ", rew_obs)
    print("Expected Reward (75)   : ", rew_75)
    print("Expected Reward (Goal) : ", rew_end)
    print("======================================")
    print("        EPS ESTIMATION")
    print("Expected Step per Epi  : ", total_step*0.5)
    print("Total Steps            : ", total_step*0.5*options.MAX_EPISODE)
    print("EPS at Last Episode    : ", options.INIT_EPS*options.EPS_DECAY**(total_step*0.5*options.MAX_EPISODE/options.EPS_ANNEAL_STEPS)  )
    print("======================================")
    print("======================================")


    return

def printSpdInfo(options):
    desiredSpd = options.INIT_SPD

    wheel_radius = 0.63407*0.5      # Wheel radius in metre

    desiredSpd_rps = desiredSpd*(1000/3600)*(1/wheel_radius)   # km/hr into radians per second

    print("Desired Speed: " + str(desiredSpd) + " km/hr = " + str(desiredSpd_rps) + " radians per seconds = " + str(math.degrees(desiredSpd_rps)) + " degrees per seconds. = " + str(desiredSpd*(1000/3600)) + "m/s" )

    return
