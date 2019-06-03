from __future__ import division
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import os
from random import choice
from time import sleep
from time import time
import tensorflow.contrib.slim as slim
import scipy.signal
import struct
from collections import defaultdict
from AC_Net import AC_Net
from DeepRMSA_Agent import DeepRMSA_Agent

# author Xiaoliang Chen, xlichen@ucdavis.edu
# copyright NGNS lab @ucdavis

# key features: uniform/nonuniform traffic distribution; window-based training; policy embedded with epsilon-greedy approach

# -----------------------------------------------------------

linkmap = defaultdict(lambda:defaultdict(lambda:None)) # Topology: NSFNet
linkmap[1][2] = (0, 1050)
linkmap[2][1] = (3, 1050)
linkmap[1][3] = (1, 1500)
linkmap[3][1] = (6, 1500)
linkmap[1][8] = (2, 2400)
linkmap[8][1] = (22, 2400)

linkmap[2][3] = (4, 600)
linkmap[3][2] = (7, 600)
linkmap[2][4] = (5, 750)
linkmap[4][2] = (9, 750)
linkmap[3][6] = (8, 1800)
linkmap[6][3] = (15, 1800)

linkmap[4][5] = (10, 600)
linkmap[5][4] = (12, 600)
linkmap[4][11] = (11, 1950)
linkmap[11][4] = (32, 1950)
linkmap[5][6] = (13, 1200)
linkmap[6][5] = (16, 1200)
linkmap[5][7] = (14, 600)
linkmap[7][5] = (19, 600)

linkmap[6][10] = (17, 1050)
linkmap[10][6] = (29, 1050)
linkmap[6][14] = (18, 1800)
linkmap[14][6] = (41, 1800)
linkmap[7][8] = (20, 750)
linkmap[8][7] = (23, 750)
linkmap[7][10] = (21, 1350)
linkmap[10][7] = (30, 1350)

linkmap[8][9] = (24, 750)
linkmap[9][8] = (25, 750)
linkmap[9][10] = (26, 750)
linkmap[10][9] = (31, 750)
linkmap[9][12] = (27, 300)
linkmap[12][9] = (35, 300)
linkmap[9][13] = (28, 300)
linkmap[13][9] = (38, 300)

linkmap[11][12] = (33, 600)
linkmap[12][11] = (36, 600)
linkmap[11][13] = (34, 750)
linkmap[13][11] = (39, 750)
linkmap[12][14] = (37, 300)
linkmap[14][12] = (42, 300)
linkmap[13][14] = (40, 150)
linkmap[14][13] = (43, 150)

nonuniform = False #True
# traffic distrition, when non-uniform traffic is considered
trafic_dis = [[0, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 1],
                      [2, 0, 2, 1, 8, 2, 1, 5, 3, 5, 1, 5, 1, 4],
                      [1, 2, 0, 2, 3, 2, 11, 20, 5, 2, 1, 1, 1, 2],
                      [1, 1, 2, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2],
                      [1, 8, 3, 1, 0, 3, 3, 7, 3, 3, 1, 5, 2, 5],
                      [4, 2, 2, 1, 3, 0, 2, 1, 2, 2, 1, 1, 1, 2],
                      [1, 1, 11, 2, 3, 2, 0, 9, 4, 20, 1, 8, 1, 4],
                      [1, 5, 20, 1, 7, 1, 9, 0, 27, 7, 2, 3, 2, 4],
                      [2, 3, 5, 2, 3, 2, 4, 27, 0, 75, 2, 9, 3, 1],
                      [1, 5, 2, 2, 3, 2, 20, 7, 75, 0, 1, 1, 2, 1],
                      [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 61],
                      [1, 5, 1, 2, 5, 1, 8, 3, 9, 1, 2, 0, 1, 81],
                      [1, 1, 1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 0, 2],
                      [1, 4, 2, 2, 5, 2, 4, 4, 0, 1, 61, 81, 2, 0]]

prob = np.array(trafic_dis)/np.sum(trafic_dis)

LINK_NUM = 44
NODE_NUM = 14

model2_flag = 0 # number of future time slots to look into

N = 10 # number of paths each src-dest pair
M = 1 # first M starting FS allocation positions are considered
k_path = 5
n_actions = k_path*M

# we do not input ttl
x_dim_p = NODE_NUM*2 + k_path*(1 + M*2 + 2 + model2_flag*3) # For each path: 1) FS needed, 2) starting index and size of M available FS-blocks, 3) average size, total available FS on the path
x_dim_v = NODE_NUM*2 + k_path*(1 + M*2 + 2 + model2_flag*3) #
num_layers = 5
layer_size = 128
regu_scalar = 1e-4

max_cpu = 16

lambda_req = 12
lambda_time = [14]
SLOT_TOTAL = 100

len_lambda_time = len(lambda_time)

gamma = 0.95 # penalty on future reward
episode_size = 1000 # number of requests in each episode
batch_size = 200 # probably smaller value, e.g., 50, would be better for higher blocking probability (see JLT)

# generate source and destination pairs
Src_Dest_Pair = []
prob_arr = []
for ii in range(NODE_NUM):
	for jj in range(NODE_NUM):
         if ii != jj:
            prob_arr.append(prob[ii][jj])
            temp = []
            temp.append(ii+1)
            temp.append(jj+1)
            Src_Dest_Pair.append(temp)
num_src_dest_pair = len(Src_Dest_Pair)
prob_arr[-1] += 1 - sum(prob_arr)

Candidate_Paths = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:None)))       #Candidate_Paths[i][j][k]:the k-th path from i to j
fp = open('Src_Dst_Paths.dat','rb')
for ii in range(1,NODE_NUM*NODE_NUM+1):#NODE_NUM*NODE_NUM import precalculated paths (in terms of path_links)
#    temp_path = []
    if ii%NODE_NUM == 0:
        i = ii//NODE_NUM
        j = (ii%NODE_NUM) + NODE_NUM
    else:
        i = (ii//NODE_NUM) + 1
        j = ii%NODE_NUM

    temp_num = []
    for tt in range(N):
        temp_num += list (struct.unpack("i"*1,fp.read(4*1)))# temp_num[0]: the node-num of path k
	
    if i != j:
        for k in range(N):
            temp_path = list(struct.unpack("i"*temp_num[k],fp.read(4*temp_num[k])))
            Candidate_Paths[i][j][k] = temp_path # note, if there are less than N paths for certain src-dest pairs, then the last a few values of temp_num equate to '0'
fp.close()
# -----------------------------------------------------------

load_model = False#True
model_path = 'model'

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype = tf.int32, name = 'global_episodes', trainable = False)
    trainer = tf.train.AdamOptimizer(learning_rate = 1e-5)
    #trainer = tf.train.RMSPropOptimizer(learning_rate = 1e-5, decay = 0.99, epsilon = 0.0001)
    master_network = AC_Net(scope = 'global',
                            trainer = None,
                            x_dim_p = x_dim_p,
                            x_dim_v = x_dim_v,
                            n_actions = n_actions,
                            num_layers = num_layers,
                            layer_size = layer_size,
                            regu_scalar = regu_scalar) # Generate global network
    num_agents = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    if num_agents > max_cpu:
        num_agents = max_cpu # as most assign max_cpu CPUs
    agents = []
    # Create worker classes
    for i in range(num_agents):
        agents.append(DeepRMSA_Agent(i,trainer,linkmap,LINK_NUM,NODE_NUM,SLOT_TOTAL,k_path,M,lambda_req,lambda_time,len_lambda_time,gamma,episode_size,batch_size,Src_Dest_Pair,Candidate_Paths,num_src_dest_pair,model_path,global_episodes,regu_scalar,x_dim_p,x_dim_v,n_actions,num_layers,layer_size,model2_flag,nonuniform,prob_arr))
    saver = tf.train.Saver(max_to_keep = 5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # Start the "rmsa" process for each agent in a separate threat.
    agent_threads = []
    for agent in agents:
        agent_rmsa = lambda: agent.rmsa(sess, coord, saver)
        t = threading.Thread(target=(agent_rmsa))
        t.start()
        sleep(0.5)
        agent_threads.append(t)
    coord.join(agent_threads)