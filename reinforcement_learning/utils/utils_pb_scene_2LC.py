import numpy as np
import pybullet as p
import pybullet_data
import random
import os
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
#   direction : 0/1/2, left/mid,right
#   scene_const
#   openWall : T/F
# Output
#   goal_id : id of the goal point
#   wall_handle_list : array of wall's handle


def createTee(x_pos, y_pos, direction, scene_const, openWall=True):
    wall_h = 0.5

    wall_handle_list = []

    # Walls left & right
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_len, wall_h], [
                            x_pos - scene_const.lane_width*0.5, y_pos, 0]))      # left
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_len, wall_h], [
                            x_pos + scene_const.lane_width*0.5, y_pos, 0]))      # right

    # Walls front & back
    wall_handle_list.append(createWall([0.5*scene_const.turn_len, 0.02, wall_h], [
                            x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width, 0]))               # front
    wall_handle_list.append(createWall([0.5*scene_const.lane_width, 0.02, wall_h], [
                            x_pos, y_pos - 1*scene_const.lane_len*0.5, 0]))              # back

    # Walls at intersection
    wall_len = 0.5*(scene_const.turn_len - scene_const.lane_width)
    wall_handle_list.append(createWall([0.5*wall_len, 0.02, wall_h], [
                            x_pos + 0.5*scene_const.lane_width + 0.5*wall_len, y_pos + scene_const.lane_len*0.5, 0]))
    wall_handle_list.append(createWall([0.5*wall_len, 0.02, wall_h], [
                            x_pos - 0.5*scene_const.lane_width - 0.5*wall_len, y_pos + scene_const.lane_len*0.5, 0]))

    # Walls at the end
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_width, wall_h], [
                            x_pos - 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0]))
    wall_handle_list.append(createWall([0.02, 0.5*scene_const.lane_width, wall_h], [
                            x_pos + 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0]))

    # Create Obstacle
    obs_width = scene_const.obs_w * scene_const.lane_width
    wall_handle_list.append(createWall([0.5*obs_width, 0.5*obs_width, wall_h], [x_pos + 0.5*(
        1-scene_const.obs_w)*scene_const.lane_width, scene_const.lane_len*0.3, 0]))      # right

    # Create Goal point
    goal_z = 1.0
    if direction == 0:
        goal_id = createGoal(0.1, [x_pos - scene_const.turn_len*0.5 + 1.0,
                                   y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z])
    elif direction == 1:
        goal_id = createGoal(
            0.1, [x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z])
    elif direction == 2:
        goal_id = createGoal(0.1, [x_pos + scene_const.turn_len*0.5 - 1.0,
                                   y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z])
    else:
        raise ValueError('Undefined direction')

    return goal_id, wall_handle_list

# Generate the scene
# Input
#   genVehicle     : T/F. If true, generate vehicle, if not don't generate vehicle
#   reset_case_list: list of testcase numbers which we generate the scene
# Output
#   handle_dict

def genScene(scene_const, options, handle_dict, reset_case_list, genVehicle=True):
    if len(reset_case_list) == 0:
        return handle_dict

    if options.VEH_COUNT % options.X_COUNT != 0:
        raise ValueError('VEH_COUNT=' + str(options.VEH_COUNT) + ' must be multiple of options.X_COUNT=' + str(options.X_COUNT))

    # Generate test cases & vehicles
    for j in range(int(options.VEH_COUNT/options.X_COUNT)):
        for i in range(options.X_COUNT):
            # print("Creating x:", i, " y:", j)

            # Unrolled index
            u_index = options.X_COUNT*j + i

            if u_index in reset_case_list:
                # Generate T-intersection for now
                handle_dict['dummy'][u_index], handle_dict['wall'][u_index] = createTee( i*scene_const.case_x, j*scene_const.case_y, i % 3, scene_const)

                # Save vehicle handle
                if genVehicle == True:
                    temp = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"), basePosition=[ i*scene_const.case_x, j*scene_const.case_y + scene_const.veh_init_y, 0], globalScaling=scene_const.veh_scale, useFixedBase=False, baseOrientation=[0, 0, 0.707, 0.707])
                    handle_dict['vehicle'][u_index] = temp

                # Resets all wheels
                for wheel in range(p.getNumJoints(handle_dict['vehicle'][u_index])):
                    # print("joint[",wheel,"]=", p.getJointInfo(vehicle_handle[u_index],wheel))
                    p.setJointMotorControl2(handle_dict['vehicle'][u_index], wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                    p.getJointInfo(handle_dict['vehicle'][u_index], wheel)

    return handle_dict

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


def initScene(scene_const, options, veh_index_list, handle_dict, randomize=False):
    # If reset list is empty, just return
    if len(veh_index_list) == 0:
        return

    vehicle_handle = handle_dict['vehicle']
    steer_handle = handle_dict['steer']
    motor_handle = handle_dict['motor']
    case_wall_handle = handle_dict['wall']
    # obs_handle     = handle_dict['obstacle']

    for veh_index in veh_index_list:
        # Reset position of vehicle. Randomize x-position if enabled
        x_index = veh_index % options.X_COUNT
        y_index = int(veh_index / options.X_COUNT)

        if randomize == False:
            p.resetBasePositionAndOrientation(vehicle_handle[veh_index], [ scene_const.case_x * x_index, scene_const.case_y * y_index + scene_const.veh_init_y, 0], [0, 0, 0.707, 0.707])
        else:
            x_pos = random.uniform(-1*scene_const.lane_width *
                                   0.5 + 0.6, scene_const.lane_width*0.5 - 0.6)
            p.resetBasePositionAndOrientation(vehicle_handle[veh_index], [
                                              scene_const.case_x * x_index + x_pos, scene_const.case_y * y_index + scene_const.veh_init_y, 0], [0, 0, 0.707, 0.707])

        # Reset steering & motor speed. TODO: Perhaps do this with setMotorArray to do it in a single line?
        for s in steer_handle:
            p.resetJointState(vehicle_handle[veh_index], s, targetValue=0)

        for m in motor_handle:
            p.resetJointState(
                vehicle_handle[veh_index], m, targetValue=0, targetVelocity=options.INIT_SPD)

        # Randomize the obstacle
        if randomize == True:
            obs_width = scene_const.obs_w * scene_const.lane_width
            temp_prob = random.uniform(0, 1)
            p.removeBody(case_wall_handle[veh_index][8])
            if temp_prob > (2/3):
                # right
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x + 0.5*( 1-scene_const.obs_w)*scene_const.lane_width, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])       # right
            elif temp_prob > (1/3):
                # middle
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])       # right
            else:
                # left
                case_wall_handle[veh_index][8] = createWall([0.5*obs_width, 0.5*obs_width, scene_const.wall_h], [x_index*scene_const.case_x - 0.5*( 1-scene_const.obs_w)*scene_const.lane_width, scene_const.case_y * y_index + scene_const.lane_len*0.3, 0])      # right


        # Randomize the goal point                
        if randomize == True:
            # Define varialbes 
            dummy_handle    = handle_dict['dummy']
            goal_z          = 1.0        
            x_pos           = random.uniform(-1*scene_const.turn_len*0.5 + 1.0, scene_const.turn_len*0.5 - 1.0)

            # Remove existing goal point
            p.removeBody( dummy_handle[veh_index] )

            # Create new goal point
            dummy_handle[veh_index] = createGoal(0.1, [x_pos + x_index * scene_const.case_x, y_index * scene_const.case_y + 0.5*scene_const.lane_len + 0.5*scene_const.lane_width, goal_z])
    return
