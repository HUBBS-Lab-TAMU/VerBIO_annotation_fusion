import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
from scipy.signal import savgol_filter
from scipy.stats import pearsonr,spearmanr, norm
import os
import torch
from torch import nn
from torch import optim
import scipy.special as sp
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
    
def expectation_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c,sigma,alpha):
    sigma_const=False
    for pid in data_dict.keys():
        
        annotator_id_list = list(data_dict[pid].keys())
        
        a_star_arr = torch.tensor(a_star[pid], requires_grad=True)
        best_arr = a_star_arr.detach().numpy()
        T = len(a_star_arr)
        optimizer = optim.SGD([a_star_arr], lr=0.001)
        
        min_loss = np.inf
        count_iter = 0
        lval = []
        save_iter=800
        for opt_iter in range(200):
            loss_opt = torch.tensor(0.0, dtype=float,requires_grad=True)
           
            optimizer.zero_grad()
            
            for n in range(len(annotator_id_list)):
                
                sigma_val = torch.tensor(sigma[n], dtype=float)
                alpha_val = torch.tensor(alpha[n], dtype=float)
                T = len(a_star_arr)
                
                a_n = torch.tensor(data_dict[pid][annotator_id_list[n]],dtype=float)
                

                F_n = torch.tensor(make_Fn2(F_weight[:,n] , T, c),dtype=float)
                
                err_vec = a_n - torch.matmul(F_n, a_star_arr) - F_bias[n] * torch.ones(T, dtype=float)
                tmp2 = torch.matmul(err_vec,err_vec)
                if sigma_const:
                    loss_opt = loss_opt + tmp2
                else:
                    Ts_r = T*torch.log(2*np.pi*sigma_val)
                    # if abs(Ts_r) > 1500:
                    #     print(opt_iter, Ts_r, sigma_val)
                    sk_r = torch.log(torch.erfc((alpha_val/(np.sqrt(2)*sigma_val))*torch.mean(-1*err_vec)))
                    # print(torch.sum(-1*err_vec))
                    loss_opt = loss_opt + (0.5/(sigma_val)**2)*tmp2 + Ts_r - sk_r
                    # loss_opt = loss_opt + (0.5/(sigma_val)**2)*tmp2 + Ts_r
            
                
            lval.append(loss_opt)
                
                
            # print(loss_opt.grad_fn)
            # print(loss_opt.grad_fn.next_functions)
            loss_opt.backward()
            
            optimizer.step()
            
            current_loss = loss_opt.detach().numpy()
            # a_star_arr = torch.clamp(a_star_arr, min=0, max=1)
            
            
            with torch.no_grad():
                a_star_arr[:] = a_star_arr.clamp(0, 1)
                
            if np.isnan(current_loss):
                break
            
            if current_loss >= min_loss:
                count_iter += 1
            else:
                min_loss = current_loss
                count_iter = 0
                best_arr = a_star_arr.detach().numpy()
                save_iter = opt_iter+1
                
                
            if count_iter == 10:
                break
            
            if opt_iter>0 and (lval[-2]-lval[-1])/lval[-2] < 0.00001:
                break
            
                                  
            # print(a_star_arr.grad)
        # a_star[pid] = best_arr
        a_star[pid] = savgol_filter(best_arr, 5, 3) 
        # print(save_iter)
        # if count_iter == 15:
        #     a_star[pid] = best_arr
        # else:
        #     a_star[pid] = a_star_arr.detach().numpy()
        # if count_iter== 15:
        #     print(lval)
            
        # plt.plot(lval)
        # plt.title(pid)
        # plt.show()      

    return a_star


def maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c, sigma,window_size,alpha):
    sigma_const = False
    for n in range(len(annotator_id_list)):
        
        d_n = torch.tensor(F_weight[:,n], dtype=float, requires_grad=True)
        d_b = torch.tensor(F_bias[n], dtype=float, requires_grad=True)
        sigma_val = torch.tensor(sigma[n], dtype=float, requires_grad=True)
        alpha_val = torch.tensor(alpha[n], dtype=float, requires_grad=True)
        
        
        sigma_best = sigma_val.detach().numpy()
        alpha_best = alpha_val.detach().numpy()
        F_weight_best = d_n.detach().numpy()
        F_bias_best = d_b.detach().numpy()
        min_loss = np.inf
        count_iter = 0

        optimizer = optim.Adam([d_n, d_b, sigma_val, alpha_val], lr=0.005)
        # optimizer = optim.Adam([d_n, d_b, sigma_val], lr=0.005)
        # optimizer = optim.SGD([d_n, d_b, sigma_val], lr=0.05)
        
        lval = []
        save_iter=200
        for opt_iter in range(100):
            loss_opt = torch.tensor(0.0, dtype=float,requires_grad=True)
            optimizer.zero_grad()
            for pid in data_dict.keys():
                a_star_arr = a_star[pid]
                T = len(a_star_arr)
                A = make_A2(a_star_arr, window_size,c)
                A_s = torch.tensor(A, dtype=float)
                
                a_n = torch.tensor(data_dict[pid][annotator_id_list[n]],dtype=float)
                
                err_vec = a_n - torch.matmul(A_s, d_n) - d_b*torch.ones(T, dtype=float)
                tmp2 = torch.matmul(err_vec,err_vec)
                if sigma_const:
                    loss_opt = loss_opt + tmp2
                else:
                    Ts_r = T*torch.log(2*np.pi*sigma_val)
                    # if abs(Ts_r) > 1500:
                    #     print(opt_iter, Ts_r, sigma_val)
                    sk_r = torch.log(torch.erfc((alpha_val/(np.sqrt(2)*sigma_val))*torch.mean(-1*err_vec)))
                    # print(torch.sum(-1*err_vec))
                    loss_opt = loss_opt + (0.5/(sigma_val)**2)*tmp2 + Ts_r - sk_r
                    # loss_opt = loss_opt + (0.5/(sigma_val)**2)*tmp2 + Ts_r
            
            # print('loss', loss_opt)
            # loss_opt = torch.sum(ind)
            
            
            
            lval.append(loss_opt)

        # print(loss_opt.grad_fn)
        # print(loss_opt.grad_fn.next_functions)
            loss_opt.backward()
            
            optimizer.step()
            # print('grad')
            # print(d_n.grad)
            # print(d_b.grad)
            # print(sigma_val.grad)
            # print(alpha_val.grad)
            # print('end')
            
            current_loss = loss_opt.detach().numpy()
            
            if np.isnan(current_loss):
                break
            
            
            if current_loss >= min_loss:
                count_iter += 1
            else:
                min_loss = current_loss
                count_iter = 0
                sigma_best = sigma_val.detach().numpy()
                alpha_best = alpha_val.detach().numpy()
                F_weight_best = d_n.detach().numpy()
                F_bias_best = d_b.detach().numpy()
                save_iter = opt_iter+1
                
            if count_iter == 15:
                break
            
            # if (opt_iter > 0) and ((lval[-2]-lval[-1])/lval[-2])
            

        # print('count', count_iter)
        sigma[n] = sigma_best
        alpha[n] = alpha_best
        F_weight[:,n] = F_weight_best
        F_bias[n] = F_bias_best
        # if count_iter == 15:
        #     sigma[n] = sigma_best
        #     # alpha[n] = alpha_best
        #     F_weight[:,n] = F_weight_best
        #     F_bias[n] = F_bias_best
        # else:
        #     sigma[n] = sigma_val.detach().numpy()
        #     # alpha[n] = alpha_val.detach().numpy()
        #     F_weight[:,n] = d_n.detach().numpy()
        #     F_bias[n] = d_b.detach().numpy()
        
        # print(save_iter)
        # plt.plot(lval)
        # plt.title(annotator_id_list[n])
        # plt.show() 

    return F_weight, F_bias, sigma, alpha
    
    

def calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg, sigma, sigma_const, c,alpha):

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
                    sk_i = np.log(sp.erfc((alpha[n]/(np.sqrt(2)*sigma[n]))*np.mean(-1*err_vec)))
                    # sk_i=0
                    sum_indiv = sum_indiv + T*math.log(2*np.pi*sigma[n]) + (0.5/(sigma[n])**2)*np.matmul(np.transpose(err_vec), err_vec)-sk_i
        
        loss_val = loss_val + sum_indiv
        # loss_val = loss_val + sum_indiv
#    print(loss_val)   
    return loss_val


def expectation_step_test(a_star, data_dict, annotator_id_list, F_weight, F_bias, c,sigma,alpha):
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

def norm_rating(rating):
    return (rating-1)/4



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
        
        if init_const:  
            a_star[pid] = (r1+r2+r4+r5)/4
            
        else:
            a_star[pid] = np.random.choice([1,2,3,4,5], t)
            # a_star[pid] = np.random.rand(t)
        
        rating_dict = data_dict[pid]
        # sc.append(sample_diff2(rating_dict,rl))
        
    return data_dict, a_star

def main(sdname):
    # c = 4
    init_const = True
    l2_reg = False
    sigma_const = False
    maximize_start = True
    
    output_path = './fusion_output/SkewNormal'
    for ws in range(2, 8):     ### past/future: range(2,8), both range(ws_val, ws_val+1)
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
        max_iter = 50
        tol_val = 0.0005
        eps = 1e-15
        
        #### Creates dataset dictionary
        print('Reading Data')
        #### Creates dataset dictionary
        
        data_dict, a_star = datareader(data_dir_train,init_const)
        
        
        
        ### Filter weight bias initialization
        
        if init_const:
            F_weight = np.zeros((window_size, num_annotator))
            F_weight[c, :]=1
        
            F_bias = np.array([0.0, 0.0, 0.0, 0.0])
            
            sigma = np.ones(num_annotator,)
            alpha = -2*np.ones(num_annotator,)
        
        else:
            F_weight = np.random.rand(window_size, num_annotator)
            F_bias = np.random.rand(num_annotator)
        
            sigma = np.ones(num_annotator,)
            alpha = np.ones(num_annotator,)
            
        if maximize_start:
        #### Maximization Step
            F_weight, F_bias, sigma, alpha = maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias, c, sigma, window_size,alpha)
        
        
        print('Fusing Training labels')
        
        loss_arr = []
        end_loop = False
        
        for iterval in range(max_iter):
            print(f'Iteration: {iterval+1}')
            ##### Expectation Step
            print('Expectation')
            a_star = expectation_step(a_star, data_dict, annotator_id_list, F_weight, F_bias,c,sigma,alpha)
        
            #### Maximization Step
            print('Maximization')
            F_weight, F_bias, sigma, alpha = maximization_step(a_star, data_dict, annotator_id_list, F_weight, F_bias,c, sigma, window_size,alpha)
           
            ##### Calculate Log-likelihood
            loss_val = calc_log_likelihood(data_dict, a_star, F_weight, F_bias, l2_reg,sigma, sigma_const,c, alpha)
            loss_arr.append(loss_val)
            print(f'Loss:{loss_val}')
            # print(F_weight, F_bias, sigma)
            if iterval == 0:
                old_loss = loss_val
            else:
                new_loss = loss_val
                # print(old_loss, new_loss)
                if ((old_loss - new_loss)/old_loss < tol_val) or (new_loss>old_loss) or np.isnan(new_loss):
                    end_loop = True
                else:
                    best_param = [a_star, F_weight, F_bias, sigma, alpha]
                    best_iter = iterval+1
                old_loss = new_loss
            
            
            if end_loop:
                break
        
            # for pid in data_dict.keys():
            #     a_star_arr = a_star[pid]
            #     r1 = data_dict[pid]['R1']
            #     r2 = data_dict[pid]['R2']
            #     r4 = data_dict[pid]['R4']
            #     r5 = data_dict[pid]['R5']
                
            #     r_mean = (r1+r2+r4+r5)/4
            #     t = np.arange(len(r_mean))
            # #    plt.plot(t,a_star_arr)
            # #    plt.show()
            # #    plt.plot(t, r_mean)
            # #    plt.show()
            # #    break
            
            #     fig = plt.figure()
            #     ax0 = fig.add_subplot(2, 1,1)
                    
                    
            # #    plt.xlabel('Time (seconds)')
            # #    plt.ylabel('Rating')
                
            #     ax0.plot(t, r_mean,linewidth=4)
            #     ax0.legend(['Average Rating'])   
            #     ax1 = fig.add_subplot(2, 1,2)
            #     fig.suptitle(pid, fontsize=16)
            
            #     ax1.plot(t, a_star_arr,linewidth=4)
            
                
                
            #     ax0.set_xlabel('Time (seconds)')
            #     ax0.set_ylabel('Rating')
            
            #     ax1.legend(['Calculated Rating'])
            #     ax1.set_xlabel('Time (seconds)')
            #     ax1.set_ylabel('Rating')
                
            #     # plt.show()
            
                    
                    
                    
            #     os.makedirs(f'./EM_sknorm/i{iterval+1}',exist_ok=True)     
                    
            #     nm = f'./EM_sknorm/i{iterval+1}/{pid}.png'
            #     fig.savefig(nm,bbox_inches = 'tight')
            #     fig.clf()
                
                    
        fig_dir = f'{output_dir}/figs/train'
        os.makedirs(fig_dir,exist_ok=True)
        
        plt.plot(loss_arr)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        
        plt.savefig(f'{output_dir}/loss.png', bbox_inches = 'tight')
        # plt.show()
        plt.clf()
        print(f'best iteration: {best_iter}')
        
        print('saving')
        
        
        # plt.savefig('loss_sknorm.png', bbox_inches = 'tight')
        
        a_star, F_weight, F_bias, sigma, alpha = best_param
        
        
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
        
            
        
        param_dict = {'W': F_weight, 'b':F_bias, 'sigma': sigma, 'alpha':alpha, 'loss':loss_arr}
        
        with open(f'{output_dir}/all_param.pickle', 'wb') as handle:
            pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
        print('Reading data')
            
        data_dict, a_star = datareader(data_dir_test, init_const)
        
        F_weight = param_dict['W']
        F_bias = param_dict['b']
        sigma = param_dict['sigma']
        alpha = param_dict['alpha']
        
        
        print('Fusing Test labels')
        a_star = expectation_step_test(a_star, data_dict, annotator_id_list, F_weight, F_bias, c,sigma,alpha)
        
        
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