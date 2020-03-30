import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# This is class for storing all the data

# This will also includes various functionalities such as saving into pikcle for making a graph

# Helpers
# Calculate the running average
# x: input 1D numpy array
# N: Window
def rolling_window(a, window):
    shape = np.asarray(a).shape[:-1] + (np.asarray(a).shape[-1] - window + 1, window)
    strides = np.asarray(a).strides + (np.asarray(a).strides[-1],)
    return np.lib.stride_tricks.as_strided(np.asarray(a), shape=shape, strides=strides)



# Class for storing and plotting data
#   epi_reward : reward of each episode
#   avg_loss   : loss of every update. Average across the batch 
#   eps        : eps used for each episode 
class data_pack:
    def __init__(self, start_time_str, epi_reward_data = None, avg_loss_data = None, eps_data = None, success_rate_data = None):
        # Data
        if epi_reward_data == None:
            self.epi_reward         = []       # List of rewards of single episode
        else:
            self.epi_reward         = epi_reward_data
        
        if avg_loss_data == None:
            self.avg_loss           = [ 0 ]       # list of loss. per step(?)
        else:
            self.avg_loss           = avg_loss_data

        if avg_loss_data == None:
            self.eps                = []       # list of loss. per step(?)
        else:
            self.eps                = avg_loss_data

        if success_rate_data == None:
            self.success_rate       = []
        else:
            self.success_rate       = success_rate_data

        # File Names
        self.start_time_str     = start_time_str

    # Update reward array
    def add_reward(self, new_reward ):
        self.epi_reward.append( new_reward ) 
        return

    # Update loss
    def add_loss(self, new_loss):
        self.avg_loss.append( new_loss )
        return

    # Update loss
    def add_eps(self, new_eps):
        self.eps.append( new_eps )
        return

    # Update success rate
    def add_success_rate(self, new_rate):
        self.success_rate.append( new_rate )
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

    def plot_loss(self, save_path = './result_data/avg_loss_value_data/', title = '' ):
        # Plot Average Step Loss
        plt.figure(1)
        fig, ax2 = plt.subplots()
        ax2.plot(self.avg_loss)
        
        ax2.set_title( title + "Average Loss per Batch Step")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Avg Loss")
        fig.savefig( save_path + 'avg_loss_value_data_' + self.start_time_str + '.png') 
        return


    # Save a graph of reward, epsilon and success rate
    def plot_reward(self, running_avg, save_path = './result_data/reward_data/', title = ''):
        plt.figure(0)
        fig, ax1 = plt.subplots()

        # x coord & rolling window
        x_coord  = range(0,len(self.epi_reward) - running_avg + 1)
        roll     = rolling_window(self.epi_reward,running_avg)
        roll_eps = rolling_window(self.eps,running_avg)
        roll_rate= rolling_window(self.success_rate,running_avg)

        # print('roll:', roll)
        # print('mean:', np.mean(roll,-1))
        # print('std:', np.std(roll,-1))

        # Mean
        ax1.plot(x_coord, np.mean(roll,-1))
        # Std Dev
        ax1.fill_between(x_coord, np.mean(roll,-1) + np.std(roll,-1), np.mean(roll,-1) - np.std(roll,-1), alpha = 0.5)

        ax1.set_title( title + "Running Average of Episode Reward (Window:" + str(running_avg) + ')')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward (Running Average)")

        # Second y-axis for reward
        ax2 = ax1.twinx()
        # EPS
        ax2.plot(x_coord, np.mean(roll_eps,-1), color='red', label='eps')
        # Rate
        ax2.plot(x_coord, np.mean(roll_rate,-1), color='green')

        ax2.set_ylabel('epsilon & success rate')
        ax2.set_ylim(0,1)

        # Save fig
        fig.savefig( save_path + 'reward_data_' + self.start_time_str + '_' + 'ra' + str(running_avg) + '.png')
 
        return
