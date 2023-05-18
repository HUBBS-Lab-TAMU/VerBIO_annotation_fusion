import numpy as np
import pandas as pd
import itertools
from scipy.stats import pearsonr, spearmanr

def agreement_metric(df,comb_list, thresh):
    
    count = 0
    for combo in comb_list:
        r1 = df[combo[0]]
        r2 = df[combo[1]]
        
        corr, pval = spearmanr(r1, r2)

        if corr >= thresh:
            count += 1
            
    if count == 0:
        return False
    else:
        return True


####### This part of code checks the number of sessions for different threshold        

thresh_corr_list = np.arange(0.3, 0.65, 0.05)
thresh_ratio_list = np.arange(0.5, 0.95, 0.05)


output_list = []
for thresh_corr in thresh_corr_list:
    for thresh_ratio in thresh_ratio_list:

        annotator_id_list = ['R1', 'R2', 'R4', 'R5']
        
        comb_list = [list(x) for x in itertools.combinations(annotator_id_list, 2)]
        session_list = ['PRE', 'TEST01', 'TEST02', 'TEST03', 'TEST04',
                        'TEST05', 'TEST06', 'TEST07', 'TEST08', 'POST']
                        
        data_dir = '../VerBIO_data'
        q_dict = {}
        pid_dict = {}
        ratio_dict = {}
        for session_id in session_list:
            print(session_id)
            for pid in range(1,74):
                full_pid = f'P{str(pid).zfill(3)}'
                fid = f'{data_dir}/{session_id}/Annotation/{session_id}_{full_pid}_annotation.xlsx'
                
                if session_id == 'TEST02' and full_pid == 'P067':
                    continue
                
                try:
                    df = pd.read_excel(fid)
                    
                    qc = agreement_metric(df,comb_list, thresh_corr)
                    
                    if full_pid in pid_dict:
                        pid_dict[full_pid] += 1
                        if qc:
                            q_dict[full_pid] += 1
                    else:
                        pid_dict[full_pid] = 1
                        if qc:
                            q_dict[full_pid] = 1
                        else:
                            q_dict[full_pid] = 0
                            
                    # else:
                    #     if full_pid in pid_dict:
                    #         pid_dict[full_pid] += 1
                    #     else:
                    #         pid_dict[full_pid] = 1
                        
                except:
                    continue
            # break



        session_train = 0
        pid_train = 0
        pid_list = []
        for full_pid in pid_dict.keys():
            
            ratio_dict[full_pid] = q_dict[full_pid]/pid_dict[full_pid]
            
            if ratio_dict[full_pid] >= thresh_ratio:
                session_train += pid_dict[full_pid]
                pid_train += 1
                pid_list.append(full_pid)
        
        output_list.append([thresh_corr, thresh_ratio, session_train, pid_train, pid_list])
            
        print(thresh_corr, thresh_ratio, session_train, pid_train, pid_list)  
        

output_df = pd.DataFrame(output_list, columns = ['Correlation Threshold', 'Ratio Threshold', 'Session', 'PID', 'PID list'])
    


######### This part of code does the train - test split based on choices made from the threshold


##### Raw Annotation


# data_dir = '../VerBIO_data'
# output_dir = './data_split/raw_annotation'
# session_list = ['PRE', 'TEST01', 'TEST02', 'TEST03', 'TEST04',
#                 'TEST05', 'TEST06', 'TEST07', 'TEST08', 'POST']

# train_pid_list = ['P001', 'P003', 'P004', 'P006', 'P008', 'P009', 'P011', 'P013', 
#                   'P017', 'P018', 'P020', 'P021', 'P026', 'P027', 'P032', 'P038',
#                   'P039', 'P040', 'P043', 'P044', 'P051', 'P053', 'P056', 'P057', 
#                   'P060', 'P061', 'P062', 'P063', 'P065', 'P066', 'P071', 'P073']


# for session_id in session_list:
#     print(session_id)
#     for pid in range(1,74):
#         full_pid = f'P{str(pid).zfill(3)}'
#         fid = f'{data_dir}/{session_id}/Annotation/{session_id}_{full_pid}_annotation.xlsx'
        
#         try:
#             df = pd.read_excel(fid)
                
#             if full_pid in train_pid_list:
#                 output_fid = f'{output_dir}/training_set/{full_pid}_{session_id}_annotation.xlsx'
#                 df.to_excel(output_fid, index=False)
            
#             else:
#                 output_fid = f'{output_dir}/test_set/{full_pid}_{session_id}_annotation.xlsx'
#                 df.to_excel(output_fid, index=False)
        
#         except:
#             continue
        
        
# ###### eGemaps feature

# data_dir = "../Feature Extraction 2/Audio_segmentedFeature"
# output_dir = './data_split/feature'
# session_list = ['PRE', 'TEST01', 'TEST02', 'TEST03', 'TEST04',
#                 'TEST05', 'TEST06', 'TEST07', 'TEST08', 'POST']

# train_pid_list = ['P001', 'P003', 'P004', 'P006', 'P008', 'P009', 'P011', 'P013', 
#                   'P017', 'P018', 'P020', 'P021', 'P026', 'P027', 'P032', 'P038',
#                   'P039', 'P040', 'P043', 'P044', 'P051', 'P053', 'P056', 'P057', 
#                   'P060', 'P061', 'P062', 'P063', 'P065', 'P066', 'P071', 'P073']


# for session_id in session_list:
#     print(session_id)
#     for pid in range(1,74):
#         full_pid = f'P{str(pid).zfill(3)}'
#         fid = f'{data_dir}/{session_id}/1sec/{session_id}_{full_pid}_eGemaps.xlsx'
        
#         try:
#             df = pd.read_excel(fid)
                
#             if full_pid in train_pid_list:
#                 output_fid = f'{output_dir}/training_set/{full_pid}_{session_id}_eGemaps.xlsx'
#                 df.to_excel(output_fid, index=False)
            
#             else:
#                 output_fid = f'{output_dir}/test_set/{full_pid}_{session_id}_eGemaps.xlsx'
#                 df.to_excel(output_fid, index=False)
        
#         except:
#             continue
