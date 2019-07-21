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

# in v1_4, we start to use continuous time simulation (still with a time granularity, but instead of the number of requests at each following the Poisson distribution,
# the inter-arrival time between requests follows the negative exponential distribution)
# therefore, we can remove the use flag_map from the input

# consider only 16 slots

# for load3, we set lambda_req = 8, for load 2, 5, for load 1, 3

# change topology, here below
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

nonuniform = False #True#
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
SLOT_TOTAL = 100

N = 10 # number of paths each src-dest pair
M = 1 # first M starting FS allocation positions are considered

#
kpath = 5 # = 1 SP-FF, = 5, KSP-FF

lambda_req = 12 # average number of requests per provisioning period, for uniform traffic, = 10, for nonuniform traffic = 16
# lambda_time = [5+2*x for x in range(6)] # average service time per request; randomly select one value from the list for each episode evaluated
lambda_time = [14] # 25 for all jlt experiments
len_lambda_time = len(lambda_time)

# generate source and destination pairs
# for each src-dst pair, we calculate its cumlative propability based on the traffic distribution
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
#print(Candidate_Paths)

def _get_path (src,dst,Candidate_Paths,k): # get path k of from src->dst
    if src == dst:
        print('error: _get_path()')
        path = []
    else:
        path = Candidate_Paths[src][dst][k]
        if path is None:
            return None
    return path
	
def calclink(p):  # map path to links
    path_link = []
    for a,b in zip(p[:-1],p[1:]):
        k = linkmap[a][b][0]
        path_link.append(k)
    return path_link
	
def get_new_slot_temp(slot_temp, path_link, slot_map):

    for i in path_link:
        for j in range(SLOT_TOTAL):
            slot_temp[j] = slot_map[i][j] & slot_temp[j]
    return slot_temp

# only used when we apply heuristic algorithms
def mark_vector(vector,default):
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
	
def judge_availability(slot_temp,current_slots,FS_id):
    (flag, slotflag, slotscontinue) = mark_vector(slot_temp,1)
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
	
def update_slot_map_for_committing_wp(slot_map, current_wp_link, current_fs, current_fe, slot_map_t, current_TTL): # update slotmap, mark allocated FS' as occupied
    for ll in current_wp_link:
        for s in range(current_fs,current_fe + 1):
            if slot_map[ll][s] != 1 or slot_map_t[ll][s] != 0: #means error
                print('Error--update_slot_map_for_committing_wp!')
            else: # still unused
                slot_map[ll][s] = 0
                slot_map_t[ll][s] = current_TTL
    return slot_map, slot_map_t
	
def update_slot_map_for_releasing_wp(slot_map, current_wp_link, current_fs, current_fe): # update slotmap, mark released FS' as free
    for ll in current_wp_link:
        for s in range(current_fs,current_fe + 1):
            if slot_map[ll][s] != 0: # this FS should be occupied by current request, !=0 means available now, which is wrong
                print('Error--update_slot_map_for_releasing_wp!')
            else: # still unused
                slot_map[ll][s] = 1
    return slot_map
	
def release(slot_map, request_set, slot_map_t, time_to):  # update slotmap to release FS' occupied by expired requests
    if request_set:
        # update slot_map_t
        for ii in range(LINK_NUM):
            for jj in range(SLOT_TOTAL):
                if slot_map_t[ii][jj] > time_to:
                    slot_map_t[ii][jj] -= time_to
                elif slot_map_t[ii][jj] > 0:
                    slot_map_t[ii][jj] = 0
                    
        #
        del_id = []
        for rr in request_set:
            request_set[rr][3] -= time_to # request_set[rr][3] is TTL
            if request_set[rr][3] <= 0:
                current_wp_link = request_set[rr][0]
                fs_wp = request_set[rr][1]
                fe_wp = request_set[rr][2]
                # release slots on the working path of the request
                slot_map = update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp, fe_wp)
                del_id.append(rr)
        for ii in del_id:
            del request_set[ii]
    return  slot_map, request_set, slot_map_t

def cal_len(path):
    path_len = 0
    for a,b in zip(path[:-1],path[1:]):
        path_len += linkmap[a][b][1]
    return path_len
	
def cal_FS(bandwidth,path_len):
    if path_len <= 625:
        num_FS = math.ceil(current_bandwidth/(4*12.5))+1 # 1 as guard band FS
    elif path_len <= 1250:
        num_FS = math.ceil(current_bandwidth/(3*12.5))+1   
    elif path_len <= 2500:
        num_FS = math.ceil(current_bandwidth/(2*12.5))+1
    else:
        num_FS = math.ceil(current_bandwidth/(1*12.5))+1
    return int(num_FS)
    
if __name__ == "__main__":

    bp_arr = []
    for ex in range(10):
        # initiate the EON
        slot_map = [[1 for x in range(SLOT_TOTAL)] for y in range(LINK_NUM)]  # Initialized to be all available
        slot_map_t = [[0 for x in range(SLOT_TOTAL)] for y in range(LINK_NUM)]  # the time each FS will be occupied

        service_time = lambda_time[np.random.randint(0, len_lambda_time)]
        lambda_intervals = 1 / lambda_req  # average time interval between request

        request_set = {}

        req_id = 0
        num_blocks = 0

        time_to = 0
        num_req_measure = 10000
        resource_util = []

        while req_id < num_req_measure + 3000:

            (slot_map, request_set, slot_map_t) = release(slot_map, request_set, slot_map_t, time_to)

            time_to = 0
            while time_to == 0:
                time_to = np.random.exponential(lambda_intervals)

            if True:  # If is used just for the sake of convenience...

                req_id += 1

                # generate current request
                if nonuniform is True:
                    sd_onehot = [x for x in range(num_src_dest_pair)]
                    sd_id = np.random.choice(sd_onehot, p=prob_arr)
                    temp = Src_Dest_Pair[sd_id]
                else:
                    temp = Src_Dest_Pair[np.random.randint(0, num_src_dest_pair)]
                current_src = temp[0]
                current_dst = temp[1]
                current_bandwidth = np.random.randint(25, 101)
                current_TTL = 0
                while current_TTL == 0 or current_TTL >= service_time * 2:
                    current_TTL = np.random.exponential(service_time)

                #  start provision the request
                blocking = 0

                for rr in range(kpath):

                    path_id = rr // M  # path to use
                    FS_id = math.fmod(rr, M)  # the FS_id's available FS-block to use

                    path = _get_path(current_src, current_dst, Candidate_Paths, path_id)
                    path_len = cal_len(path)  # physical length of the path
                    num_FS = cal_FS(current_bandwidth, path_len)
                    slot_temp = [1] * SLOT_TOTAL
                    path_links = calclink(path)
                    slot_temp = get_new_slot_temp(slot_temp, path_links,
                                                  slot_map)  # spectrum utilization on the whole path
                    (flag, fs_start, fs_end) = judge_availability(slot_temp, num_FS, FS_id)
                    if flag == 1:
                        slot_map, slot_map_t = update_slot_map_for_committing_wp(slot_map, path_links, fs_start, fs_end,
                                                                                 slot_map_t,
                                                                                 current_TTL)  # update slotmap
                        temp_ = []  # update in-service requests
                        temp_.append(list(path_links))
                        temp_.append(fs_start)
                        temp_.append(fs_end)
                        temp_.append(current_TTL)
                        request_set[req_id] = temp_
                        break
                    elif rr == kpath - 1:
                        blocking = 1

                if req_id > 3000:
                    num_blocks += blocking  # count the number of requests that are blocked
                    resource_util.append(1-np.sum(slot_map)/(LINK_NUM*SLOT_TOTAL))

        bp = num_blocks / num_req_measure
        bp_arr.append(bp)
        print('Blocking Probability = ', np.mean(bp_arr))
        print('Mean Resource Utilization =', np.mean(resource_util))