import numpy as np
import math
import sys
import matplotlib
import matplotlib.pyplot as plt
import time
import pickle
from argparse import ArgumentParser
import os.path



###########################################################
# Plot data obtained from basic_dqn.py
###########################################################

# PARSE INPUTS
parser = ArgumentParser(
    description='File for learning'
    )
parser.add_argument('--VEH_POS', type=str, help='path to vehicle position data')
parser.add_argument('--VEH_RATE', type=int, help='sampling rate for vehicle', default=100 )
parser.add_argument('--END_EPI', type=int, help='last epsidoe to plot')
parser.add_argument('--REW', type=str, help='path to reward data')
options = parser.parse_args()



#################################
# GRAPH VEHICLE POSITION
#################################

if os.path.isfile(options.VEH_POS):
    # Load
    veh_data = pickle.load( open( options.VEH_POS, "rb" ) )

    # Check option
    if options.END_EPI == None or options.END_EPI > veh_data.shape[0]:
        options.END_EPI = veh_data.shape[0]

    
    # Plot
    plt.figure(0)

    for i in range(0, options.END_EPI, options.VEH_RATE):           # For each episode
#        plt.scatter(veh_data[i][:,0], veh_data[i][:,1], label = "Trail #" + str(i)) 
        plt.plot(veh_data[i][:,0], veh_data[i][:,1], '-o', label = "Trail #" + str(i)) 

    # Plot Map
    plt.plot([1.25, 1.25],[0, 60],'k')
    plt.plot([-3.75, -3.75],[0, 60],'k')
    plt.plot([-1.25, -1.25],[0, 60],'k')
    plt.plot([0, 0],[0, 60],'k--')
    plt.plot([-2.5, -2.5],[0, 60],'k--')
    plt.plot([0],[60],'xr') 

    # Plot properties
    plt.axis( (-3.75, 1.25, 0, 80)   )
    plt.legend()
    plt.title("Vehicle Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()



#################################
# GRAPH REWARD
#################################
if os.path.isfile(options.REW):
    # Load
    reward_data = pickle.load( open( options.REW, "rb" ) )

    # Calculate running average
    avg_steps = 100     # num of episodes for taking average 
    avg_epi_reward_data = np.zeros(reward_data.size)
   
    for i in range(0,avg_epi_reward_data.size):
        if i >= avg_steps:
            avg_epi_reward_data[i] = np.mean(reward_data[i-avg_steps+1:i])  
    

    # Plot
    plt.figure(1)

    plt.plot(avg_epi_reward_data)
    plt.title("Average Reward. Steps:" + str(avg_steps))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
