import numpy as np
import pybullet as p
import pybullet_data
import random
import os

#########################################
# Files for generating the scene
#########################################
# Create Wall
# Input
#   size : [x,y,z]
#   pos  : [x,y,z]
def createWall( size, pos ):
    temp = p.createMultiBody( baseMass = 0, baseCollisionShapeIndex = p.createCollisionShape( p.GEOM_BOX, halfExtents = size ), baseVisualShapeIndex = p.createVisualShape( shapeType = p.GEOM_BOX, rgbaColor = [1,1,1,1], halfExtents = size), basePosition = pos )
    if temp < 0:
        raise ValueError("Failed to create multibody at createWall")
    else: 
        return temp

# Create Goal Point
# Input
#  size : radius, single number
# Output
#  goal_point id
def createGoal( size, pos ):
    temp = p.createMultiBody( baseMass = 0, baseCollisionShapeIndex = p.createCollisionShape( p.GEOM_SPHERE, radius = size ), baseVisualShapeIndex = p.createVisualShape( shapeType = p.GEOM_SPHERE, rgbaColor = [1,0,0,1], radius = size), basePosition = pos )
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
def createTee( x_pos, y_pos, direction, scene_const, openWall = True ): 
    wall_h = 0.5

    wall_handle_list = []

    # Walls left & right
    wall_handle_list.append( createWall( [0.02, 0.5*scene_const.lane_len, wall_h], [x_pos - scene_const.lane_width*0.5, y_pos, 0] ) )      # left
    wall_handle_list.append( createWall( [0.02, 0.5*scene_const.lane_len, wall_h], [x_pos + scene_const.lane_width*0.5, y_pos, 0] ) )      # right

    # Walls front & back
    wall_handle_list.append( createWall( [0.5*scene_const.turn_len, 0.02, wall_h], [x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width,0] ) )               # front
    wall_handle_list.append( createWall( [0.5*scene_const.lane_width, 0.02, wall_h], [x_pos, y_pos -1*scene_const.lane_len*0.5,0] ) )              # back

    # Walls at intersection
    wall_len = 0.5*( scene_const.turn_len - scene_const.lane_width ) 
    wall_handle_list.append( createWall( [0.5*wall_len, 0.02, wall_h], [x_pos + 0.5*scene_const.lane_width + 0.5*wall_len, y_pos + scene_const.lane_len*0.5,0] ) )
    wall_handle_list.append( createWall( [0.5*wall_len, 0.02, wall_h], [x_pos - 0.5*scene_const.lane_width - 0.5*wall_len, y_pos + scene_const.lane_len*0.5,0] ) )

    # Walls at the end
    wall_handle_list.append( createWall( [0.02, 0.5*scene_const.lane_width, wall_h], [x_pos - 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0] ) )
    wall_handle_list.append( createWall( [0.02, 0.5*scene_const.lane_width, wall_h], [x_pos + 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0] ) )




    # Create Goal point
    goal_z = 1.0
    if direction == 0:
        goal_id = createGoal( 0.1, [x_pos - scene_const.turn_len*0.5 + 0.5, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z] )
    elif direction == 1:
        goal_id = createGoal( 0.1, [x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z] )
    elif direction == 2:
        goal_id = createGoal( 0.1, [x_pos + scene_const.turn_len*0.5 - 0.5, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, goal_z] )
    else:
        raise ValueError('Undefined direction')

    return goal_id, wall_handle_list

# Generate the scene
#
# Output
#   vehicle_handle : array of vehicle handle
#   dummy_handle   : array of handle to goal point
#   wall_case_handle : 2-d array, each row consists of handles to all objects forming a single testcase

def genScene( scene_const, options ):
    # Handles that will be returned
    vehicle_handle      = np.zeros(options.VEH_COUNT, dtype=int)
    dummy_handle        = np.zeros(options.VEH_COUNT, dtype=int)
    case_wall_handle    = np.zeros((options.VEH_COUNT,scene_const.wall_cnt), dtype=int) 

    # Load plane
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane100.urdf"))

    if options.VEH_COUNT % options.X_COUNT != 0:
        raise ValueError('VEH_COUNT=' + str(options.VEH_COUNT) + ' must be multiple of options.X_COUNT=' + str(options.X_COUNT))

    # Generate test cases & vehicles
    for j in range(int(options.VEH_COUNT/options.X_COUNT)):
        for i in range( options.X_COUNT ):
            print("Creating x:",i," y:", j)

            # Unrolled index
            u_index = options.X_COUNT*j + i

            # Generate T-intersection for now
            dummy_handle[ u_index ], case_wall_handle[ u_index ]  = createTee( i*scene_const.case_x, j*scene_const.case_y, i % 3, scene_const )

            # Save vehicle handle
            temp = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"), basePosition = [i*scene_const.case_x,j*scene_const.case_y + scene_const.veh_init_y,0], globalScaling = 2.0, useFixedBase = False, baseOrientation = [0,0,0.707,0.707])
            vehicle_handle[ u_index ] = temp

            # Resets all wheels
            for wheel in range(p.getNumJoints(vehicle_handle[u_index])):
                # print("joint[",wheel,"]=", p.getJointInfo(vehicle_handle[u_index],wheel))
                p.setJointMotorControl2(vehicle_handle[u_index],wheel,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
                p.getJointInfo(vehicle_handle[u_index],wheel)	

    return vehicle_handle, dummy_handle, case_wall_handle


# Initialize test cases
# Input
def initScene( scene_const, options, veh_index_list, handle_dict, randomize = False):
    # If reset list is empty, just return
    if len(veh_index_list) == 0:
        return

    vehicle_handle = handle_dict['vehicle']
    steer_handle   = handle_dict['steer']
    motor_handle   = handle_dict['motor']
    # obs_handle     = handle_dict['obstacle']
    sensor_handle  = handle_dict['sensor']

    for veh_index in veh_index_list:
        # Reset position of vehicle. Randomize x-position if enabled
        x_index = veh_index % options.X_COUNT
        y_index = int( veh_index / options.X_COUNT ) 

        if randomize == False:
            p.resetBasePositionAndOrientation( vehicle_handle[veh_index],[ scene_const.case_x * x_index, scene_const.case_y * y_index + scene_const.veh_init_y,0], [0,0,0.707,0.707]  )
        else:
            x_pos = random.uniform(-1*scene_const.lane_width*0.5 + 1.5,scene_const.lane_width*0.5 - 1.5)
            p.resetBasePositionAndOrientation( vehicle_handle[veh_index],[ scene_const.case_x * x_index + x_pos, scene_const.case_y * y_index + scene_const.veh_init_y,0], [0,0,0.707,0.707]  )

        # Reset steering & motor speed. TODO: Perhaps do this with setMotorArray to do it in a single line?
        for s in steer_handle:
            p.resetJointState( vehicle_handle[veh_index], s, targetValue = 0)
        for m in motor_handle:
            p.resetJointState( vehicle_handle[veh_index], m, targetValue = 0, targetVelocity = options.INIT_SPD)
        # setMotorPosition(scene_const.clientID, handle_dict['steer'], np.zeros(options.VEH_COUNT))

        # Reset motor speed
        # setMotorSpeed(scene_const.clientID, motor_handle[veh_index], options.INIT_SPD)

        # Set initial speed of vehicle
        # p.resetBaseVelocity( vehicle_handle[veh_index], linearVelocity = [0, options.INIT_SPD, 0] )
        p.resetBaseVelocity( vehicle_handle[veh_index], linearVelocity = [0, 0, 0] )

        # Reset position of obstacle
        # if randomize == True:
        #     if random.random() > 0.5:
                # p.resetBasePositionAndOrientation( vehicle_handle[veh_index],[ scene_const.case_x * x_index + x_pos, scene_const.case_y * y_index + scene_const.veh_init_y,0], [0,0,0.707,0.707]  )
        #     else:
                # p.resetBasePositionAndOrientation( vehicle_handle[veh_index],[ scene_const.case_x * x_index + x_pos, scene_const.case_y * y_index + scene_const.veh_init_y,0], [0,0,0.707,0.707]  )
        
        # Reset position of dummy    
        # if randomize == False:
        #     pass
        # else:
        #     pass

    # Reset Steering
    # p.setJointMotorControlArray( vehicle_handle, motor_handle, p.POSITION_CONTROL, targetPosition = np.zeros(options.VEH_COUNT) )

    # Reset Motor Speed
    return













