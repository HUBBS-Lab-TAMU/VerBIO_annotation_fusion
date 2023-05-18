# import subprocess, shlex
# import numpy as np
# import pandas as pd
# import os

# seed_list = [6732, 8360, 9953, 3872, 3441, 3008, 8069, 42, 3577,  845, 4157, 1371,3392, 7438, 7084, 5636,  183, 4014, 5855, 100]
# seed_list = [3577,  845, 4157, 1371,3392, 7438, 7084, 5636,  183, 4014, 5855, 100]
# # label_list = ['Mean', 'EWE', 'DBA', 'RAAW', 'Selective']    ### Baseline

# # ### Normal
# # label_list = ['w2_c1_Normal', 'w3_c2_Normal', 'w4_c3_Normal', 'w5_c4_Normal',
# #        'w6_c5_Normal', 'w7_c6_Normal', 'w2_c0_Normal', 'w3_c0_Normal',
# #        'w4_c0_Normal', 'w5_c0_Normal', 'w6_c0_Normal', 'w7_c0_Normal',
# #        'w10_c3_Normal', 'w10_c4_Normal', 'w10_c5_Normal', 'w10_c6_Normal',
# #        'w11_c4_Normal', 'w11_c5_Normal', 'w11_c6_Normal', 'w12_c5_Normal',
# #        'w12_c6_Normal', 'w13_c6_Normal', 'w3_c1_Normal', 'w4_c1_Normal',
# #        'w4_c2_Normal', 'w5_c1_Normal', 'w5_c2_Normal', 'w5_c3_Normal',
# #        'w6_c1_Normal', 'w6_c2_Normal', 'w6_c3_Normal', 'w6_c4_Normal',
# #        'w7_c1_Normal', 'w7_c2_Normal', 'w7_c3_Normal', 'w7_c4_Normal',
# #        'w7_c5_Normal', 'w8_c1_Normal', 'w8_c2_Normal', 'w8_c3_Normal',
# #        'w8_c4_Normal', 'w8_c5_Normal', 'w8_c6_Normal', 'w9_c2_Normal',
# #        'w9_c3_Normal', 'w9_c4_Normal', 'w9_c5_Normal', 'w9_c6_Normal']


# ### SkewNormal
# label_list = ['w2_c1_SkewNormal', 'w3_c2_SkewNormal', 'w4_c3_SkewNormal',
#        'w5_c4_SkewNormal', 'w6_c5_SkewNormal', 'w7_c6_SkewNormal',
#        'w2_c0_SkewNormal', 'w3_c0_SkewNormal', 'w4_c0_SkewNormal',
#        'w5_c0_SkewNormal', 'w6_c0_SkewNormal', 'w7_c0_SkewNormal',
#        'w10_c3_SkewNormal', 'w10_c4_SkewNormal', 'w10_c5_SkewNormal',
#        'w10_c6_SkewNormal', 'w11_c4_SkewNormal', 'w11_c5_SkewNormal',
#        'w11_c6_SkewNormal', 'w12_c5_SkewNormal', 'w12_c6_SkewNormal',
#        'w13_c6_SkewNormal', 'w3_c1_SkewNormal', 'w4_c1_SkewNormal',
#        'w4_c2_SkewNormal', 'w5_c1_SkewNormal', 'w5_c2_SkewNormal',
#        'w5_c3_SkewNormal', 'w6_c1_SkewNormal', 'w6_c2_SkewNormal',
#        'w6_c3_SkewNormal', 'w6_c4_SkewNormal', 'w7_c1_SkewNormal',
#        'w7_c2_SkewNormal', 'w7_c3_SkewNormal', 'w7_c4_SkewNormal',
#        'w7_c5_SkewNormal', 'w8_c1_SkewNormal', 'w8_c2_SkewNormal',
#        'w8_c3_SkewNormal', 'w8_c4_SkewNormal', 'w8_c5_SkewNormal',
#        'w8_c6_SkewNormal', 'w9_c2_SkewNormal', 'w9_c3_SkewNormal',
#        'w9_c4_SkewNormal', 'w9_c5_SkewNormal', 'w9_c6_SkewNormal']

# for seed_val in seed_list:
#     for nm in label_list:
#         command = f'python main_sn.py --task stress --lr 0.0002 --rnn_n_layers 4 --epochs 200 --label_type {nm} --label_folder Mean --seed {seed_val} --use_gpu'

#         call_params = shlex.split(command)
#         subprocess.call(call_params)


#     for nm in label_list:
#         command = f'python main_sn.py --task stress --lr 0.0002 --rnn_n_layers 4 --epochs 200 --label_type {nm} --label_folder Selective --seed {seed_val} --use_gpu'
            
#         call_params = shlex.split(command)
#         subprocess.call(call_params)

import subprocess, shlex
import numpy as np
import pandas as pd
import os


seed_list = [6732, 8360, 9953, 3872, 3441, 3008, 8069, 42, 3577,  845, 4157, 1371,3392, 7438, 7084, 5636,  183, 4014, 5855, 100]
seed_list = [100]
# label_list = ['Mean', 'EWE', 'DBA', 'RAAW', 'Selective']    ### Baseline

### Normal
label_list = ['w2_c1_Normal', 'w3_c2_Normal', 'w4_c3_Normal', 'w5_c4_Normal',
       'w6_c5_Normal', 'w7_c6_Normal', 'w2_c0_Normal', 'w3_c0_Normal',
       'w4_c0_Normal', 'w5_c0_Normal', 'w6_c0_Normal', 'w7_c0_Normal',
       'w10_c3_Normal', 'w10_c4_Normal', 'w10_c5_Normal', 'w10_c6_Normal',
       'w11_c4_Normal', 'w11_c5_Normal', 'w11_c6_Normal', 'w12_c5_Normal',
       'w12_c6_Normal', 'w13_c6_Normal', 'w3_c1_Normal', 'w4_c1_Normal',
       'w4_c2_Normal', 'w5_c1_Normal', 'w5_c2_Normal', 'w5_c3_Normal',
       'w6_c1_Normal', 'w6_c2_Normal', 'w6_c3_Normal', 'w6_c4_Normal',
       'w7_c1_Normal', 'w7_c2_Normal', 'w7_c3_Normal', 'w7_c4_Normal',
       'w7_c5_Normal', 'w8_c1_Normal', 'w8_c2_Normal', 'w8_c3_Normal',
       'w8_c4_Normal', 'w8_c5_Normal', 'w8_c6_Normal', 'w9_c2_Normal',
       'w9_c3_Normal', 'w9_c4_Normal', 'w9_c5_Normal', 'w9_c6_Normal']


# ### SkewNormal
# label_list = ['w2_c1_SkewNormal', 'w3_c2_SkewNormal', 'w4_c3_SkewNormal',
#        'w5_c4_SkewNormal', 'w6_c5_SkewNormal', 'w7_c6_SkewNormal',
#        'w2_c0_SkewNormal', 'w3_c0_SkewNormal', 'w4_c0_SkewNormal',
#        'w5_c0_SkewNormal', 'w6_c0_SkewNormal', 'w7_c0_SkewNormal',
#        'w10_c3_SkewNormal', 'w10_c4_SkewNormal', 'w10_c5_SkewNormal',
#        'w10_c6_SkewNormal', 'w11_c4_SkewNormal', 'w11_c5_SkewNormal',
#        'w11_c6_SkewNormal', 'w12_c5_SkewNormal', 'w12_c6_SkewNormal',
#        'w13_c6_SkewNormal', 'w3_c1_SkewNormal', 'w4_c1_SkewNormal',
#        'w4_c2_SkewNormal', 'w5_c1_SkewNormal', 'w5_c2_SkewNormal',
#        'w5_c3_SkewNormal', 'w6_c1_SkewNormal', 'w6_c2_SkewNormal',
#        'w6_c3_SkewNormal', 'w6_c4_SkewNormal', 'w7_c1_SkewNormal',
#        'w7_c2_SkewNormal', 'w7_c3_SkewNormal', 'w7_c4_SkewNormal',
#        'w7_c5_SkewNormal', 'w8_c1_SkewNormal', 'w8_c2_SkewNormal',
#        'w8_c3_SkewNormal', 'w8_c4_SkewNormal', 'w8_c5_SkewNormal',
#        'w8_c6_SkewNormal', 'w9_c2_SkewNormal', 'w9_c3_SkewNormal',
#        'w9_c4_SkewNormal', 'w9_c5_SkewNormal', 'w9_c6_SkewNormal']


for seed_val in seed_list:
    for nm in label_list:
        command = f'python main2.py --task stress --lr 0.0002 --rnn_n_layers 4 --epochs 200 --label_type {nm} --label_folder Mean --seed {seed_val} --use_gpu'

        call_params = shlex.split(command)
        subprocess.call(call_params)


    for nm in label_list:
        command = f'python main2.py --task stress --lr 0.0002 --rnn_n_layers 4 --epochs 200 --label_type {nm} --label_folder Selective --seed {seed_val} --use_gpu'
            
        call_params = shlex.split(command)
        subprocess.call(call_params)