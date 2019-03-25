import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import pickle

# This is class for storing all the data

# This will also includes various functionalities such as saving into pikcle for making a graph

# Helpers
# Calculate the running average
# x: input 1D numpy array
# N: Window
def runningAverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class data_pack:
    def __init__(self, start_time_str):
        # Data
        self.epi_reward         = np.zeros( 0 )       # List of rewards of single episode

        # File Names
        self.start_time_str     = start_time_str

    # Update reward array
    def add_reward(self, new_reward ):
        np.append( self.epi_reward, new_reward ) 

        return


    # Save reward data into a pickle
    def save_reward(self):
        # Save Reward Data
        outfile = open( './result_data/reward_data/reward_data_' + self.start_time_str, 'wb')  
        pickle.dump( self.epi_reward, outfile )
        outfile.close()
        
    # Save a graph of reward
    def plot_reward(self, running_avg):
        plt.figure(0)
        fig, ax1 = plt.subplots()
        ax1.plot(runningAverage(self.epi_reward,running_avg))

        ax1.set_title("Running Average of Episode Reward")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward (Running Average)")
        fig.savefig('./result_data/reward_data/reward_data_' + self.start_time_str + '_' + 'ra' + str(running_avg) + '.png')
 
        return
