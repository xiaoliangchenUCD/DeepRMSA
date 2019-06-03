from __future__ import division
from collections import defaultdict
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime
import tensorflow.contrib.slim as slim
from AC_Net import AC_Net
import threading
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
from collections import defaultdict

class DeepRMSA_Agent():
    
    def __init__(self,
                 id,
                 trainer,
                 linkmap,
                 LINK_NUM,
                 NODE_NUM,
                 SLOT_TOTAL,
                 k_path,
                 M,
                 lambda_req,
                 lambda_time,
                 len_lambda_time,
                 gamma,
                 episode_size,
                 batch_size,
                 Src_Dest_Pair,
                 Candidate_Paths,
                 num_src_dest_pair,
                 model_path,
                 global_episodes,
                 regu_scalar,
                 x_dim_p,
                 x_dim_v,
                 n_actions,
                 num_layers,
                 layer_size,
                 model2_flag,
                 nonuniform,
                 prob_arr):
        self.name = 'agent_' + str(id)
        self.trainer = trainer
        self.linkmap = linkmap
        self.LINK_NUM = LINK_NUM
        self.NODE_NUM = NODE_NUM
        self.SLOT_TOTAL = SLOT_TOTAL
        self.k_path = k_path
        self.M = M
        self.lambda_req = lambda_req
        self.lambda_time = lambda_time
        self.len_lambda_time = len_lambda_time
        self.gamma = gamma
        self.episode_size = episode_size
        self.batch_size = batch_size
        self.Src_Dest_Pair = Src_Dest_Pair
        self.Candidate_Paths = Candidate_Paths
        self.num_src_dest_pair = num_src_dest_pair
        self.model_path = model_path
        self.model2_flag = model2_flag
        self.nonuniform = nonuniform
        self.prob_arr = prob_arr
        self.global_episodes = global_episodes #
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + self.name)
        #
        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        self.local_network = AC_Net(scope = self.name,
                                    trainer = self.trainer,
                                    x_dim_p = self.x_dim_p,
                                    x_dim_v = self.x_dim_v,
                                    n_actions = self.n_actions,
                                    num_layers = num_layers,
                                    layer_size = layer_size,
                                    regu_scalar = regu_scalar)
        self.update_local_ops = self.update_target_graph('global',self.name)
        #
        self.slot_map = [[1 for x in range(self.SLOT_TOTAL)] for y in range(self.LINK_NUM)] #Initialized to be all available
        self.slot_map_t = [[0 for x in range(self.SLOT_TOTAL)] for y in range(self.LINK_NUM)] # the time each FS will be occupied
        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1/self.lambda_req # average time interval between request
        self.request_set = {}
        self.his_slotmap = []
        # self.all_ones = [[1 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        # self.all_negones = [[0 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        
    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def _get_path (self, src,dst,Candidate_Paths,k): # get path k of from src->dst
        if src == dst:
            print('error: _get_path()')
            path = []
        else:
            path = Candidate_Paths[src][dst][k]
            if path is None:
                return None
        return path
	
    def calclink(self, p):  # map path to links
        path_link = []
        for a,b in zip(p[:-1],p[1:]):
            k = self.linkmap[a][b][0]
            path_link.append(k)
        return path_link
	
    def get_new_slot_temp(self, slot_temp, path_link, slot_map):
        for i in path_link:
            for j in range(self.SLOT_TOTAL):
                slot_temp[j] = slot_map[i][j] & slot_temp[j]
        return slot_temp

    def mark_vector(self, vector,default):
        le = len(vector)
        flag = 0
        slotscontinue = []
        slotflag = []

        ii = 0
        while ii <= le-1: 
            tempvector = vector[ii:le]
            default_counts = tempvector.count(default)
            if default_counts == 0:
                break
            else:
                a = tempvector.index(default)
                ii += a
                flag += 1
                slotflag.append(ii)
                m = vector[ii+1:le]
                m_counts = m.count(1-default)
                if m_counts != 0:
                    n = m.index(1-default)
                    slotcontinue = n+1
                    slotscontinue.append(slotcontinue)
                    ii += slotcontinue
                else:
                    slotscontinue.append(le-ii)
                    break
        return flag, slotflag, slotscontinue
	
    def judge_availability(self, slot_temp,current_slots,FS_id):
        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp,1)
        fs = -1
        fe = -1
        if flag>0:
            n = len(slotscontinue)
            flag_availability = 0 # Initialized to be unavailable
            t = 0
            for i in range(n):
                if slotscontinue[i] >= current_slots:
                    if t == FS_id:
                        fs = slotflag[i]
                        fe = slotflag[i] + current_slots - 1
                        flag_availability = 1
                        return flag_availability, fs, fe
                    t += 1
            return flag_availability, fs, fe
        else:
            flag_availability = 0
        return flag_availability, fs, fe
	
    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t, current_TTL): # update slotmap, mark allocated FS' as occupied
        for ll in current_wp_link:
            for s in range(current_fs,current_fe + 1):
                assert slot_map[ll][s] == 1
                '''assert slot_map_t[ll][s] == 0'''
                slot_map[ll][s] = 0
                '''slot_map_t[ll][s] = current_TTL'''
        return slot_map, slot_map_t
	
    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs, current_fe): # update slotmap, mark released FS' as free
        for ll in current_wp_link:
            for s in range(current_fs,current_fe + 1):
                assert slot_map[ll][s] == 0
                slot_map[ll][s] = 1
        return slot_map
	
    def release(self, slot_map, request_set, slot_map_t, time_to):  # update slotmap to release FS' occupied by expired requests
        if request_set:
            # update slot_map_t
            '''for ii in range(self.LINK_NUM):
                for jj in range(self.SLOT_TOTAL):
                    if slot_map_t[ii][jj] > time_to:
                        slot_map_t[ii][jj] -= time_to
                    elif slot_map_t[ii][jj] > 0:
                        slot_map_t[ii][jj] = 0'''           
            #
            del_id = []
            for rr in request_set:
                request_set[rr][3] -= time_to # request_set[rr][3] is TTL
                if request_set[rr][3] <= 0:
                    current_wp_link = request_set[rr][0]
                    fs_wp = request_set[rr][1]
                    fe_wp = request_set[rr][2]
                    # release slots on the working path of the request
                    slot_map = self.update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp, fe_wp)
                    del_id.append(rr)
            for ii in del_id:
                del request_set[ii]
        return  slot_map, request_set, slot_map_t

    def cal_len(self, path):
        path_len = 0
        for a,b in zip(path[:-1],path[1:]):
            path_len += self.linkmap[a][b][1]
        return path_len
	
    def cal_FS(self, bandwidth, path_len):
        if path_len <= 625:
            num_FS = math.ceil(bandwidth/(4*12.5))+1 # 1 as guard band FS
        elif path_len <= 1250:
            num_FS = math.ceil(bandwidth/(3*12.5))+1
        elif path_len <= 2500:
            num_FS = math.ceil(bandwidth/(2*12.5))+1
        else:
            num_FS = math.ceil(bandwidth/(1*12.5))+1
        return int(num_FS)
    
    # Discounting function used to calculate discounted returns.
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
    
    def train(self, espisode_buff, sess, value_est):
        espisode_buff = np.array(espisode_buff)
        input_p = espisode_buff[:self.batch_size,0]
        input_v = espisode_buff[:self.batch_size,1]
        actions = espisode_buff[:self.batch_size,2]
        rewards = espisode_buff[:,3]
        values = espisode_buff[:,4]
        
        self.rewards_plus = np.asarray(rewards.tolist() + [value_est])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        discounted_rewards = np.append(discounted_rewards, 0) # --
        discounted_rewards_batch = discounted_rewards[:self.batch_size] - (self.gamma**self.batch_size)*discounted_rewards[self.batch_size:] # --
        self.value_plus = np.asarray(values.tolist() + [value_est])
        '''advantages = rewards + self.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages)
        advantages = np.append(advantages, 0) # --
        advantages = advantages[:self.batch_size] - (self.gamma**self.batch_size)*advantages[self.batch_size:] # --'''
        advantages = discounted_rewards_batch - self.value_plus[:self.batch_size]
        
        # a filtering scheme, filter out 20% largest and smallest elements from 'discounted_rewards_batch'
        '''sorted_reward = np.argsort(discounted_rewards_batch)
        input_p = input_p[sorted_reward[10:-10]]
        input_v = input_v[sorted_reward[10:-10]]
        discounted_rewards_batch = discounted_rewards_batch[sorted_reward[10:-10]]
        actions = actions[sorted_reward[10:-10]]
        advantages = advantages[sorted_reward[10:-10]]'''

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_network.target_v: discounted_rewards_batch,
            self.local_network.Input_p: np.vstack(input_p),
            self.local_network.Input_v: np.vstack(input_v),
            self.local_network.actions: actions,
            self.local_network.advantages: advantages}
        sum_value_losss, sum_policy_loss, sum_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, _, _, regu_loss_policy, regu_loss_value = sess.run([self.local_network.loss_value,
            self.local_network.loss_policy,
            self.local_network.entropy,
            self.local_network.grad_norms_policy,
            self.local_network.grad_norms_value,
            self.local_network.var_norms_policy,
            self.local_network.var_norms_value,
            self.local_network.apply_grads_policy,
            self.local_network.apply_grads_value,
            self.local_network.regu_loss_policy,
            self.local_network.regu_loss_value],
            feed_dict = feed_dict)
        return sum_value_losss / self.batch_size, sum_policy_loss / self.batch_size, sum_entropy / self.batch_size, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, regu_loss_policy / self.batch_size, regu_loss_value / self.batch_size
        
    def rmsa(self, sess, coord, saver):  
    
        time_to = 0
        req_id = 0
        
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        episode_buffer = []
        
        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]
        
        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()
        
        all_zeros = [0 for ii in range(3 + 2*self.M)]
        all_nega_ones = [-1 for ii in range(3 + 2*self.M)]
        
        # update local dnn with the global one
        sess.run(self.update_local_ops)
        
        epsilon = 1

        print ('Starting ' + self.name)
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
           
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                actionss = []
                      
                num_blocks = 0
                
                # adaptive learning rate
                '''if episode_count % 10 == 0 and episode_count != 0:
                    self.local_network.trainer._lr = np.max([1e-6, self.local_network.trainer._lr - 1e-6])'''

                # begin an episode
                resource_util = []
                while episode_step_count < self.episode_size:
                
                    req_id += 1
                    
                    (self.slot_map, self.request_set, self.slot_map_t) = self.release(self.slot_map, self.request_set, self.slot_map_t, time_to)
                    
                    time_to = 0        
                    while time_to == 0:
                        time_to = np.random.exponential(self.lambda_intervals)

                    # generate current request
                    if self.nonuniform is True:
                        sd_id = np.random.choice(sd_onehot, p = self.prob_arr)
                        temp = self.Src_Dest_Pair[sd_id]
                    else:
                        temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]
                    current_src = temp[0]
                    current_dst = temp[1]
                    current_bandwidth = np.random.randint(25, 101)
                    current_TTL = 0
                    while current_TTL == 0 or current_TTL >= self.service_time*2:
                        current_TTL = np.random.exponential(self.service_time)
                        
                    # generate features
                    '''TTL_norm = current_TTL/(2*self.service_time)'''
                    src_onehot = node_onehot[current_src-1]
                    dst_onehot = node_onehot[current_dst-1]
                    Input_feature = []
                    Input_feature += src_onehot # s
                    Input_feature += dst_onehot # d
                    '''Input_feature.append(TTL_norm)'''
                    # include features from each path
                    if self.model2_flag > 0:
                        slot_map_fur = []
                        slot_map_tmp = copy.deepcopy(self.slot_map)
                        request_set_tmp = copy.deepcopy(self.request_set)
                        slot_map_t_tmp = copy.deepcopy(self.slot_map_t)
                        for ii in range(self.model2_flag):
                            (slot_map_tmp, request_set_tmp, slot_map_t_tmp) = self.release(slot_map_tmp, request_set_tmp, slot_map_t_tmp, 5*self.lambda_intervals)
                            slot_map_fur.append(slot_map_tmp)
                    
                    for x in range(self.k_path):
                        path = self._get_path(current_src, current_dst, self.Candidate_Paths, x)
                        if len(path) == 0:
                            '''Input_feature += all_zeros'''
                            Input_feature += all_nega_ones
                            for ii in range(self.model2_flag):
                                Input_feature += [0, 0, 0]
                        else:
                            path_len = self.cal_len(path)  # physical length of the path
                            num_FS = self.cal_FS(current_bandwidth, path_len)
                            slot_temp = [1] * self.SLOT_TOTAL
                            path_links = self.calclink(path)
                            slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)  # spectrum utilization on the whole path
                            (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                            if flag == 0 or np.max(slotscontinue) < num_FS:
                                '''Input_feature += all_zeros'''
                                Input_feature += all_nega_ones
                            else:
                                '''Input_feature.append((num_FS-1)/8) # number of FS's required using this path, 2 <= num_FS <= 9'''
                                Input_feature.append((num_FS-5.5)/3.5) # normalized
                                slotscontinue_array = np.array(slotscontinue)
                                idx = np.where(slotscontinue_array >= num_FS)[0]
                                for jj in range(self.M): # for the first self.M available FS-blocks
                                    if len(idx) > jj:
                                        '''Input_feature.append(slotflag[idx[jj]]/self.SLOT_TOTAL) # starting index
                                        Input_feature.append(slotscontinue[idx[jj]]/8) # size'''
                                        Input_feature.append(2*(slotflag[idx[jj]]-0.5*self.SLOT_TOTAL)/self.SLOT_TOTAL)
                                        Input_feature.append((slotscontinue[idx[jj]]-8)/8)
                                    else:
                                        '''Input_feature += [0, 0]'''
                                        Input_feature += [-1, -1]
                                Input_feature.append(2*(sum(slotscontinue)-0.5*self.SLOT_TOTAL)/self.SLOT_TOTAL) # total available FS's
                                Input_feature.append((np.mean(slotscontinue)-4)/4) # mean size of FS-blocks
                            #--------------------------------------------------
                            for ii in range(self.model2_flag):
                                slot_temp = [1] * self.SLOT_TOTAL
                                slot_temp = self.get_new_slot_temp(slot_temp, path_links, slot_map_fur[ii])
                                (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                                if flag == 0:
                                    Input_feature += [0, 0, 0]
                                else:
                                    Input_feature.append(sum(slotscontinue)/self.SLOT_TOTAL)
                                    Input_feature.append(np.mean(slotscontinue)/8)
                                    Input_feature.append(np.max(slotscontinue)/8)
                            #--------------------------------------------------
                            
                    Input_feature = np.array(Input_feature)
                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))
                    
                    blocking = 0

                    #Take an action using probabilities from policy network output.
                    prob_dist, value, entro = sess.run([self.local_network.policy, self.local_network.value, self.local_network.entropy],
                        feed_dict = {self.local_network.Input_p: Input_feature,
                        self.local_network.Input_v: Input_feature})
                    pp = prob_dist[0]
                    assert not np.isnan(entro)
                    '''if self.name == 'agent_0':
                        print(pp, '--')'''
                    
                    if random.random() < epsilon:
                        action_id = np.random.choice(action_onehot, p = pp)
                    else:
                        action_id = np.argmax(pp)
                    
                    # shift action id to avoid that in most cases, action_id = 0 is the best
                    '''if current_src <= 2:
                        action_id_transf = action_id
                    elif current_src <= 4:
                        action_id_transf = (action_id + 1) % self.n_actions # cyclic right move, step 1
                    else:
                        action_id_transf = (action_id + 2) % self.n_actions # cyclic right move, step 2
                    path_id = action_id_transf // self.M  # path to use
                    FS_id = math.fmod(action_id_transf, self.M)'''
                    
                    path_id = action_id // self.M  # path to use
                    FS_id = math.fmod(action_id, self.M)
                    path = self._get_path(current_src, current_dst, self.Candidate_Paths, path_id)

                    actionss.append(action_id)

                    # apply the selected action
                    if len(path) == 0: # selected an invalid action
                        blocking = 1
                    else:
                        path_len = self.cal_len(path)  # physical length of the path
                        num_FS = self.cal_FS(current_bandwidth, path_len)
                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)
                        slot_temp = self.get_new_slot_temp(slot_temp, path_links, self.slot_map)  # spectrum utilization on the whole path
                        (flag, fs_start, fs_end) = self.judge_availability(slot_temp, num_FS, FS_id)
                        if flag == 1:
                            self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(self.slot_map, path_links, fs_start, fs_end, self.slot_map_t, current_TTL)  # update slotmap
                            temp_ = []  # update in-service requests
                            temp_.append(list(path_links))
                            temp_.append(fs_start)
                            temp_.append(fs_end)
                            temp_.append(current_TTL)
                            self.request_set[req_id] = temp_
                        else:
                            blocking = 1
                    
                    r_t = 1 - 2*blocking # successful, 1, blocked, -1
                    #r_t = 1 - blocking
                    num_blocks += blocking
                    
                    episode_reward += r_t
                    total_steps += 1
                    episode_step_count += 1
                    
                    resource_util.append(1-np.sum(self.slot_map)/(self.LINK_NUM*self.SLOT_TOTAL))
                    
                    if episode_count < (3000/self.episode_size): # for warm-up
                        continue
                    
                    # store experience
                    episode_buffer.append([Input_feature, Input_feature, action_id, r_t, value[0,0]])
                    episode_values.append(value[0,0])
                    
                    if len(episode_buffer) == 2*self.batch_size-1:
                        mean_value_losss, mean_policy_loss, mean_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, regu_loss_policy, regu_loss_value = self.train(episode_buffer, sess, 0.0)
                        del(episode_buffer[:self.batch_size])
                        sess.run(self.update_local_ops) # if we want to synchronize local with global every a training is performed
                        epsilon = np.max([epsilon - 1e-5, 0.05])
                        
                # end of an episode
                episode_count += 1
                
                #sess.run(self.update_local_ops) # if we want to synchronize local with global every episode is finished
                
                if episode_count <= (3000/self.episode_size): # for warm-up
                    continue
                
                bp = num_blocks/self.episode_size
                if self.name == 'agent_0':
                    print('Blocking Probability = ', bp)
                    #print('Action Distribution', actionss.count(0)/len(actionss))
                    print('Mean Resource Utilization =', np.mean(resource_util))
                    # store blocking probability
                    fp = open('BP.dat', 'a')
                    fp.write('%f\n' % bp)
                    fp.close()
                    # store value prediction
                    fp = open('value.dat', 'a')
                    fp.write('%f\n' % np.mean(episode_values))
                    fp.close()
                    # store value loss
                    fp = open('value_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_value_losss))
                    fp.close()
                    # store policy loss
                    fp = open('policy_loss.dat', 'a')
                    fp.write('%f\n' % float(mean_policy_loss))
                    fp.close()
                    # store entroy
                    fp = open('entropy.dat', 'a')
                    fp.write('%f\n' % float(mean_entropy))
                    fp.close()
                
                self.episode_blocking.append(bp)
                self.episode_rewards.append(episode_reward)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Periodically save model parameters, and summary statistics.
                sample_step = int(1000/self.episode_size)
                if episode_count % sample_step == 0 and episode_count != 0:
                    '''if episode_count % (100*sample_step) == 0 and self.name == 'agent_0':
                        saver.save(sess, self.model_path+'/model.cptk')
                        print ("Model Saved")'''

                    '''if self.name == 'agent_0':
                        mean_reward = np.mean(self.episode_rewards[-sample_step:])
                        mean_value = np.mean(self.episode_mean_values[-sample_step:])
                        mean_blocking = np.mean(self.episode_blocking[-sample_step:])
                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        summary.value.add(tag='Perf/Blocking', simple_value=float(mean_blocking))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(mean_value_losss))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(mean_policy_loss))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(mean_entropy))
                        summary.value.add(tag='Losses/Grad Norm Policy', simple_value=float(grad_norms_policy))
                        summary.value.add(tag='Losses/Grad Norm Value', simple_value=float(grad_norms_value))
                        summary.value.add(tag='Losses/Var Norm Policy', simple_value=float(var_norms_policy))
                        summary.value.add(tag='Losses/Var Norm Value', simple_value=float(var_norms_value))
                        summary.value.add(tag='Losses/Regu Loss Policy', simple_value=float(regu_loss_policy))
                        summary.value.add(tag='Losses/Regu Loss Value', simple_value=float(regu_loss_value))
                        self.summary_writer.add_summary(summary, episode_count)

                        self.summary_writer.flush()'''
                if self.name == 'agent_0':
                    sess.run(self.increment)