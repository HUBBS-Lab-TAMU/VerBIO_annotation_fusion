###### This code is same as test_conv.py except the data reading part

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
from scipy.signal import savgol_filter
from scipy.stats import pearsonr,spearmanr, norm
import os
import pickle



def make_Fn2(d_n, T, c):
    #### This function is generalized for both side windows
    w = len(d_n)
    F_n = np.zeros((T,T))
    len_l = c
    len_r = w - c - 1
    
    for i in range(T):
        temp_vec = np.zeros(len_l+T+len_r)
        temp_vec[i:i+w] = d_n
        F_n[i,:] = temp_vec[c:c+T]
        
    return F_n


def make_A2(vec, w, c):
    #### This function is generalized for both side windows
    T = len(vec)
    X = np.zeros((T,w))
    len_l = c
    len_r = w - c - 1
    
    temp_vec = np.zeros(len_l+T+len_r)
    temp_vec[c:c+T] = vec
    
    for i in range(T):
        X[i,:] = temp_vec[i:i+w]

    return X
    
        


def expectation_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c):
    for pid in data_dict.keys():
        flag_e = 0
        annotator_id_list = list(data_dict[pid].keys())
        T = len(data_dict[pid][annotator_id_list[0]])
        sum_left = np.zeros((T,T))
        sum_right = np.zeros((T,))
        
        for n in range(len(annotator_id_list)):
            # if sample_diff(data_dict[pid], annotator_id_list[n])==False:
            #     flag_e+=1
            #     continue
            a_n = data_dict[pid][annotator_id_list[n]]
            d_n = F_weight[:,n]
            d_b = F_bias[n]
            F_n = make_Fn2(d_n , T, c)
            
            bias_vec = d_b * np.ones(T)
            sum_left = sum_left + np.matmul(np.transpose(F_n), F_n)
            sum_right = sum_right + np.matmul(np.transpose(F_n), a_n) - np.matmul(np.transpose(F_n), bias_vec)
        # print(pid,flag_e)
        if flag_e != 4:
            
            if math.fabs(1.0/np.linalg.cond(sum_left)) < eps:
                # ic+=1
                sum_left += np.identity(T)
            
            # a_star[pid] = savgol_filter(np.matmul(np.linalg.inv(sum_left), sum_right), 5, 3)
            tmp_rating = savgol_filter(np.matmul(np.linalg.inv(sum_left), sum_right), 5, 3)
            a_star[pid] = np.clip(tmp_rating, 0, 1)
            # a_star[pid] = np.matmul(np.linalg.inv(sum_left), sum_right)
            
    return a_star


def norm_rating(rating):
    return (rating-1)/4


c = 4

eps = 1e-15
side_name = 'both'
tol = '5e-4'
train_dir = f'./EM_split_result/norm/training'
test_dir = f'./EM_split_result/norm/test'
os.makedirs(test_dir,exist_ok=True)

with open(f'{train_dir}/all_param.pickle', 'rb') as handle:
    param_dict = pickle.load(handle)

for ws in range(8,9):
    print(ws)
    ic=0
    
    annotator_id_list = ['R1', 'R2', 'R4', 'R5']
    session = 'PRE'
    data_dir = './data_split/raw_annotation/test_set'   ####FIXME
    window_size = ws
    num_annotator = 4
    
    #### Creates dataset dictionary
    data_dict = {}
    a_star = {}
    
    tq = np.array([0])
    
    for fid in os.listdir(data_dir):
        # print(fid)
        df = pd.read_excel(f'{data_dir}/{fid}')
        [subject_id, session_id, tmp] = fid.split('_')
        
        dict_key = f'{subject_id}_{session_id}'
        data_dict[dict_key] = {}
        
        data_dict[dict_key]['R1'] = norm_rating(df['R1'].to_numpy())
        data_dict[dict_key]['R2'] = norm_rating(df['R2'].to_numpy())
        data_dict[dict_key]['R4'] = norm_rating(df['R4'].to_numpy())
        data_dict[dict_key]['R5'] = norm_rating(df['R5'].to_numpy())

        
        
    
    rl = [list(x) for x in itertools.combinations(annotator_id_list, 2)]
    sc = []
    for pid in data_dict.keys():
        r1 = data_dict[pid]['R1']
        r2 = data_dict[pid]['R2']
        r4 = data_dict[pid]['R4']
        r5 = data_dict[pid]['R5']
        t = len(r1)
        
        
        a_star[pid] = (r1+r2+r4+r5)/4
            
        
        
        rating_dict = data_dict[pid]
        # sc.append(sample_diff2(rating_dict,rl))
    
    
    
    ### Filter weight bias initialization
    
    
    F_weight = param_dict['W']
    F_bias = param_dict['b']
    sigma = param_dict['sigma']
    
    
    
    a_star = expectation_step(a_star, data_dict, annotator_id_list, F_weight, F_bias,c)
    
    
    
    
    for pid in data_dict.keys():
        a_star_arr = a_star[pid]
        r1 = data_dict[pid]['R1']
        r2 = data_dict[pid]['R2']
        r4 = data_dict[pid]['R4']
        r5 = data_dict[pid]['R5']
        
        r_mean = (r1+r2+r4+r5)/4
        t = np.arange(len(r_mean))
    #    plt.plot(t,a_star_arr)
    #    plt.show()
    #    plt.plot(t, r_mean)
    #    plt.show()
    #    break
    
        fig = plt.figure()
        ax0 = fig.add_subplot(2, 1,1)
            
            
    #    plt.xlabel('Time (seconds)')
    #    plt.ylabel('Rating')
        
        ax0.plot(t, r_mean,linewidth=4)
        ax0.legend(['Average Rating'])   
        ax1 = fig.add_subplot(2, 1,2)
        fig.suptitle(pid, fontsize=16)
    
        ax1.plot(t, a_star_arr,linewidth=4)
    
        
        
        ax0.set_xlabel('Time (seconds)')
        ax0.set_ylabel('Rating')
    
        ax1.legend(['Calculated Rating'])
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Rating')
        
        plt.show()
    
            
            
            
            
            
        nm = f'{test_dir}/{pid}.png'
        fig.savefig(nm,bbox_inches = 'tight')
        fig.clf()
    
    for pid in a_star.keys():
        em_res = a_star[pid]
        op_df = pd.DataFrame(em_res, columns = ['EM'])
        df_dir = f'./EM_split_result'
        os.makedirs(df_dir,exist_ok=True)
        op_df.to_excel(f'{test_dir}/{pid}_EM.xlsx', index=False)
    


    