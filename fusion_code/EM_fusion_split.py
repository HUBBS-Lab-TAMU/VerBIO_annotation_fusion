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
    eps = 1e-15
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
            
            tmp_rating = savgol_filter(np.matmul(np.linalg.inv(sum_left), sum_right), 5, 3)
            a_star[pid] = np.clip(tmp_rating, 0, 1)
            # a_star[pid] = np.matmul(np.linalg.inv(sum_left), sum_right)
            
    return a_star


def maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c):
    eps = 1e-15
    l2_reg = False
    window_size = len(F_weight)
    num_annotator = len(annotator_id_list)
    sigma = np.zeros(num_annotator,)
    for n in range(len(annotator_id_list)):
        flag_c = 0
        old_d_n = F_weight[:,n]
        old_d_b = F_bias[n]
        
        new_d_b = 0
        new_d_n = np.zeros((window_size,))
        
        sum_left = np.zeros((window_size,window_size))
        sum_right = np.zeros((window_size,))
        T_sum = 0
        for pid in data_dict.keys():
            # if sample_diff(data_dict[pid], annotator_id_list[n])==False:
            #     flag_c+=1
            #     continue
                
            a_star_arr = a_star[pid]
            a_n = data_dict[pid][annotator_id_list[n]]
            T = len(a_n)
            bias_vec = old_d_b * np.ones(T)
            F_n = make_Fn2(old_d_n , T, c)
            A = make_A2(a_star_arr, window_size,c)
            sigma_val = a_n-np.matmul(F_n,a_star_arr)- bias_vec
            sigma[n] += np.matmul(np.transpose(sigma_val),sigma_val)
            ###For bias
            new_d_b = new_d_b + np.matmul(np.transpose(np.ones(T)), a_n) - np.matmul(np.transpose(np.ones(T)), np.matmul(F_n, a_star_arr))
            T_sum = T_sum + T
            
            ### For weights
            if l2_reg:
                sum_left = sum_left + np.matmul(np.transpose(A), A) + np.identity(window_size)
            else:
                sum_left = sum_left + np.matmul(np.transpose(A), A)
            sum_right = sum_right + np.matmul(np.transpose(A), a_n) - np.matmul(np.transpose(A), bias_vec)
        
        if flag_c!=53: 
            if math.fabs(1.0/np.linalg.cond(sum_left)) < eps:
                # ic+=1
                sum_left += np.identity(window_size)
             
            F_bias[n] = new_d_b/T_sum
        
            F_weight[:,n] = np.matmul(np.linalg.inv(sum_left), sum_right)
        
        sigma[n] = math.sqrt(sigma[n]/T_sum)
    return F_weight, F_bias, sigma
    
    

def calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg, sigma, sigma_const, c):

    loss_val = 0
    for pid in data_dict.keys():
        a_star_arr = a_star[pid]
        annotator_id_list = list(data_dict[pid].keys())
        
        sum_indiv = 0
        for n in range(len(annotator_id_list)):
            a_n = data_dict[pid][annotator_id_list[n]]
            T = len(a_n)
            d_n = F_weight[:,n]
            d_b = F_bias[n]
            F_n = make_Fn2(d_n , T, c)
            bias_vec = d_b * np.ones(T)
            err_vec = a_n - np.matmul(F_n, a_star_arr) - bias_vec
            # print(np.matmul(np.transpose(err_vec), err_vec))
            if l2_reg:
                l2_vec = np.matmul(np.transpose(d_n), d_n)  
                sum_indiv = sum_indiv + np.matmul(np.transpose(err_vec), err_vec) + l2_vec
            else:
                # sum_indiv = sum_indiv + FindNormPDF(np.matmul(np.transpose(err_vec), err_vec), 0, sigma[n])
                if sigma_const:
                    sum_indiv = sum_indiv + np.matmul(np.transpose(err_vec), err_vec)
                else:
                    sum_indiv = sum_indiv + T*math.log(2*np.pi*sigma[n]) + (0.5/(sigma[n])**2)*np.matmul(np.transpose(err_vec), err_vec)
        
        loss_val = loss_val + sum_indiv
        # loss_val = loss_val + sum_indiv
#    print(loss_val)   
    return loss_val

def norm_rating(rating):
    return (rating-1)/4

def expectation_step_test(a_star, data_dict, annotator_id_list, F_weight, F_bias, c):
    eps = 1e-15
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


def corr_check(rating_dict):
    rater_list = rating_dict.keys()
    rater_pair_list = [list(x) for x in itertools.combinations(rater_list, 2)]
    
    good_rater = set()
    for pair in rater_pair_list:
        rater1 = pair[0]
        rater2 = pair[1]

        r1 = rating_dict[rater1]

        r2 = rating_dict[rater2]
        rval,pval = spearmanr(r1,r2)
        
        if rval>=0.4:
            # print(rater1, rater2)
            good_rater.add(rater1)
            good_rater.add(rater2)
    
    init_rating = np.zeros(len(r1),)
    
    if len(good_rater) == 0:
        for rater in rater_list:
            init_rating = init_rating + rating_dict[rater]
            
        init_rating = init_rating/4
    else:
        for rater in good_rater:
            init_rating = init_rating + rating_dict[rater]
            
        init_rating = init_rating/len(good_rater)
    return init_rating
    
    
            


def datareader(data_dir, init_const):
    data_dict = {}
    a_star = {}
    
    
    for fid in os.listdir(data_dir):
        df = pd.read_excel(f'{data_dir}/{fid}')
        [subject_id, session_id, tmp] = fid.split('_')
        
        dict_key = f'{subject_id}_{session_id}'
        data_dict[dict_key] = {}
        
        data_dict[dict_key]['R1'] = norm_rating(df['R1'].to_numpy())
        data_dict[dict_key]['R2'] = norm_rating(df['R2'].to_numpy())
        data_dict[dict_key]['R4'] = norm_rating(df['R4'].to_numpy())
        data_dict[dict_key]['R5'] = norm_rating(df['R5'].to_numpy())
        
        
    
    
    sc = []
    for pid in data_dict.keys():
        r1 = data_dict[pid]['R1']
        r2 = data_dict[pid]['R2']
        r4 = data_dict[pid]['R4']
        r5 = data_dict[pid]['R5']
        t = len(r1)
        
        
        rating_dict = data_dict[pid]
        
        
        if init_const:  
            a_star[pid] = (r1+r2+r4+r5)/4
            # a_star[pid] = corr_check(rating_dict) ####FIXME
            
        else:
            a_star[pid] = np.random.choice([1,2,3,4,5], t)
            # a_star[pid] = np.random.rand(t)
        
        
        # sc.append(sample_diff2(rating_dict,rl))
        
    return data_dict, a_star
    
    
def main(sdname):
    # c = 4
    
    init_const = True
    l2_reg = False
    sigma_const = False
    maximize_start = False   ###FIXME
    output_path = './fusion_output/Normal_5e-6'
    
    for ws in range(2,8):        ### past/future: range(2,8), both range(ws_val, ws_val+1)
        if sdname == 'past':
            c = ws-1   ### past: ws -1 , future: 0, both: comment out
        else:
            c = 0
        output_dir = f'{output_path}/output_w{ws}_c{c}'
        os.makedirs(output_dir,exist_ok=True)
        print(f'Window size:{ws}, Center index: {c}')
        ic=0
        
        annotator_id_list = ['R1', 'R2', 'R4', 'R5']

        data_dir_train = './data_split/raw_annotation/training_set'   ####FIXME
        data_dir_test = './data_split/raw_annotation/test_set'   ####FIXME
        window_size = ws
        num_annotator = 4
        max_iter = 1000
        tol_val = 0.00005
        eps = 1e-15
        
        print('Reading Data')
        #### Creates dataset dictionary
        
        data_dict, a_star = datareader(data_dir_train,init_const)
        
        
        ### Filter weight bias initialization
        
        if init_const:
            F_weight = np.zeros((window_size, num_annotator))
            F_weight[c, :]=1
        
            F_bias = np.array([0.0, 0.0, 0.0, 0.0])
        
        else:
            F_weight = np.random.rand(window_size, num_annotator)
            F_bias = np.random.rand(num_annotator)
        
        
        
        if maximize_start:
        #### Maximization Step
            F_weight, F_bias, sigma = maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c)
        
        
        print('Fusing Training labels')
        
        loss_arr = []
        end_loop = False
        
        for iterval in range(max_iter):
            ##### Expectation Step
            a_star = expectation_step(a_star, data_dict, annotator_id_list, F_weight, F_bias,c)
        
            #### Maximization Step
            sigma = np.zeros(num_annotator,)
            F_weight, F_bias, sigma = maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias,c)
           
            ##### Calculate Log-likelihood
            loss_val = calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg,sigma, sigma_const,c)
            loss_arr.append(loss_val)
            
            if iterval == 0:
                old_loss = loss_val
            else:
                new_loss = loss_val
                
                if abs(new_loss - old_loss)/old_loss < tol_val:
                    end_loop = True
                old_loss = new_loss
            
            
            if end_loop:
                break
                
                    
        fig_dir = f'{output_dir}/figs/train'
        os.makedirs(fig_dir,exist_ok=True)
        
        plt.plot(loss_arr)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        
        plt.savefig(f'{output_dir}/loss.png', bbox_inches = 'tight')
        # plt.show()
        plt.clf()
        
        print('saving')
        
        
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
            
            # plt.show()
        
                
                
                
                
                
            nm = f'{fig_dir}/{pid}.png'
            fig.savefig(nm,bbox_inches = 'tight')
            fig.clf()
        
        
        df_dir = f'{output_dir}/fused_rating/train'
        os.makedirs(df_dir,exist_ok=True)
        
        for pid in a_star.keys():
            em_res = a_star[pid]
            op_df = pd.DataFrame(em_res, columns = ['EM'])
            
            op_df.to_excel(f'{df_dir}/{pid}_EM.xlsx', index=False)
            
        param_dict = {'W': F_weight, 'b':F_bias, 'sigma': sigma, 'loss': loss_arr}
        
        with open(f'{output_dir}/all_param.pickle', 'wb') as handle:
            pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Reading data')
            
        data_dict, a_star = datareader(data_dir_test, init_const)
        
        F_weight = param_dict['W']
        F_bias = param_dict['b']
        sigma = param_dict['sigma']
        
        
        print('Fusing Test labels')
        a_star = expectation_step_test(a_star, data_dict, annotator_id_list, F_weight, F_bias,c)
        
        
        fig_dir = f'{output_dir}/figs/test'
        os.makedirs(fig_dir,exist_ok=True)
        
        print('saving')
        
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
            
            # plt.show()
        
                
                
                
                
                
            nm = f'{fig_dir}/{pid}.png'
            fig.savefig(nm,bbox_inches = 'tight')
            fig.clf()
        
        
        df_dir = f'{output_dir}/fused_rating/test'
        os.makedirs(df_dir,exist_ok=True)
        
        for pid in a_star.keys():
            em_res = a_star[pid]
            op_df = pd.DataFrame(em_res, columns = ['EM'])
            
            op_df.to_excel(f'{df_dir}/{pid}_EM.xlsx', index=False)
        

if __name__ == "__main__":
    main('past')
    main('future')
    # past_ws = np.arange(1,7)
    # future_ws = np.arange(1,7)

    # for pw in past_ws:
    #     for fw in future_ws:
    #         ws_val = pw+fw+1
    #         c = pw
    #         main(ws_val,c)