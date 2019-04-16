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
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



# Class for storing and plotting data
#   epi_reward : reward of each episode
#   avg_loss   : loss of every update. Average across the batch 
#   eps        : eps used for each episode 
class data_pack:
    def __init__(self, start_time_str, epi_reward_data = None, avg_loss_data = None, eps_data = None):
        # Data
        if epi_reward_data == None:
            self.epi_reward         = np.zeros( 0 )       # List of rewards of single episode
        else:
            self.epi_reward         = epi_reward_data
        
        if avg_loss_data == None:
            self.avg_loss           = np.zeros( 1 )       # list of loss. per step(?)
        else:
            self.avg_loss           = avg_loss_data

        if avg_loss_data == None:
            self.eps                = np.zeros( 0 )       # list of loss. per step(?)
        else:
            self.eps                = avg_loss_data


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

    # Update loss
    def add_eps(self, new_eps):
        self.eps = np.append( self.eps, new_eps )
        return

    # Save reward data into a pickle
    def save_reward(self, save_path = './result_data/reward_data/'):
        # Save Reward Data
        outfile = open( save_path + 'reward_data_' + self.start_time_str, 'wb')  
        pickle.dump( self.epi_reward, outfile )
        outfile.close()

        return

    # Save loss data
    def save_loss(self, save_path = './result_data/loss_data/' ): 
        outfile = open( save_path + 'avg_loss_value_data_' + self.start_time_str, 'wb')  
        pickle.dump( self.avg_loss , outfile )
        outfile.close()

        return

    def plot_loss(self, save_path = './result_data/avg_loss_value_data/' ):
        # Plot Average Step Loss
        plt.figure(0)
        fig, ax2 = plt.subplots()
        ax2.plot(self.avg_loss)
        
        ax2.set_title("Average Loss per Batch Step")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Avg Loss")
        fig.savefig( save_path + 'avg_loss_value_data_' + self.start_time_str + '.png') 
        return


    # Save a graph of reward
    def plot_reward(self, running_avg, save_path = './result_data/reward_data/'):
        plt.figure(0)
        fig, ax1 = plt.subplots()

        # x coord & rolling window
        x_coord  = range(0,self.epi_reward.size - running_avg + 1)
        roll     = rolling_window(self.epi_reward,running_avg)
        roll_eps = rolling_window(self.eps,running_avg)

        # Mean
        ax1.plot(x_coord, np.mean(roll,-1))
        # Std Dev
        ax1.fill_between(x_coord, np.mean(roll,-1) + np.std(roll,-1), np.mean(roll,-1) - np.std(roll,-1), alpha = 0.5)

        ax1.set_title("Running Average of Episode Reward (Window:" + str(running_avg) + ')')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward (Running Average)")

        # Second y-axis for reward
        ax2 = ax1.twinx()
        # EPS
        #print(x_coord)
        #print(np.mean(roll_eps,-1))
        ax2.plot(x_coord, np.mean(roll_eps,-1), color='red', label='eps')
        ax2.set_ylabel('epsilon')
        ax2.set_ylim(0,1)

        # Save fig
        fig.savefig( save_path + 'reward_data_' + self.start_time_str + '_' + 'ra' + str(running_avg) + '.png')
 
        return
