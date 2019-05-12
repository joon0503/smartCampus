import numpy as np
import pybullet as p

#########################################
# Files for generating the scene
#########################################
# Create Wall
# Input
#   size : [x,y,z]
#   pos  : [x,y,z]
def createWall( size, pos ):
    return p.createMultiBody( baseMass = 0, baseCollisionShapeIndex = p.createCollisionShape( p.GEOM_BOX, halfExtents = size ), baseVisualShapeIndex = p.createVisualShape( shapeType = p.GEOM_BOX, rgbaColor = [1,1,1,1], halfExtents = size), basePosition = pos )

# Create Goal Point
# Input
#  size : radius, single number
# Output
#  goal_point id
def createGoal( size, pos ):
    return p.createMultiBody( baseMass = 0, baseCollisionShapeIndex = p.createCollisionShape( p.GEOM_SPHERE, radius = size ), baseVisualShapeIndex = p.createVisualShape( shapeType = p.GEOM_SPHERE, rgbaColor = [1,0,0,1], radius = size), basePosition = pos )

# Create T-Intersection
# Input
#   x_pos : x position of the test case
#   direction : 0/1/2, left/mid,right
#   scene_const
#   openWall : T/F
def createTee( x_pos, y_pos, direction, scene_const, openWall = True ): 
    wall_h = 2

    # Walls left & right
    createWall( [0.02, 0.5*scene_const.lane_len, wall_h], [x_pos - scene_const.lane_width*0.5, y_pos, 0] )       # left
    createWall( [0.02, 0.5*scene_const.lane_len, wall_h], [x_pos + scene_const.lane_width*0.5, y_pos, 0] )       # right

    # Walls front & back
    createWall( [0.5*scene_const.turn_len, 0.02, wall_h], [x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width,0] )                # front
    createWall( [0.5*scene_const.lane_width, 0.02, wall_h], [x_pos, y_pos -1*scene_const.lane_len*0.5,0] )              # back

    # Walls at intersection
    wall_len = 0.5*( scene_const.turn_len - scene_const.lane_width ) 
    createWall( [0.5*wall_len, 0.02, wall_h], [x_pos + 0.5*scene_const.lane_width + 0.5*wall_len, y_pos + scene_const.lane_len*0.5,0] )
    createWall( [0.5*wall_len, 0.02, wall_h], [x_pos - 0.5*scene_const.lane_width - 0.5*wall_len, y_pos + scene_const.lane_len*0.5,0] )

    # Walls at the end
    createWall( [0.02, 0.5*scene_const.lane_width, wall_h], [x_pos - 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0] )
    createWall( [0.02, 0.5*scene_const.lane_width, wall_h], [x_pos + 0.5*scene_const.turn_len, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5, 0] )




    # Create Goal point
    if direction == 0:
        goal_id = createGoal( 0.1, [x_pos - scene_const.turn_len*0.5 + 0.5, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5,2] )
    elif direction == 1:
        goal_id = createGoal( 0.1, [x_pos, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5,2] )
    elif direction == 2:
        goal_id = createGoal( 0.1, [x_pos + scene_const.turn_len*0.5 - 0.5, y_pos + scene_const.lane_len*0.5 + scene_const.lane_width*0.5,2] )
    else:
        raise ValueError('Undefined direction')

    return goal_id
