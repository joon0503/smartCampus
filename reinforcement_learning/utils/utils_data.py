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
        self.avg_loss           = np.zeros( 1 )       # list of loss. per step(?)


        # File Names
        self.start_time_str     = start_time_str

    # Update reward array
    def add_reward(self, new_reward ):
        self.epi_reward = np.append( self.epi_reward, new_reward ) 
        return

    # Update loss
    def add_loss(self, new_loss):
        self.avg_loss = np.append( self.avg_loss, new_loss )
        return


    # Save reward data into a pickle
    def save_reward(self):
        # Save Reward Data
        outfile = open( './result_data/reward_data/reward_data_' + self.start_time_str, 'wb')  
        pickle.dump( self.epi_reward, outfile )
        outfile.close()

        return

    # Save loss data
    def save_loss(self): 
        outfile = open( 'result_data/loss_data/avg_loss_value_data_' + self.start_time_str, 'wb')  
        pickle.dump( self.avg_loss , outfile )
        outfile.close()

        return

    def plot_loss(self):
        # Plot Average Step Loss
        plt.figure(0)
        fig, ax2 = plt.subplots()
        ax2.plot(self.avg_loss)
        
        ax2.set_title("Average Loss per Batch Step")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Avg Loss")
        fig.savefig('result_data/avg_loss_value_data/avg_loss_value_data_' + self.start_time_str + '.png') 
        return


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
