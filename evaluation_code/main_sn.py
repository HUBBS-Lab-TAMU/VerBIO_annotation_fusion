import argparse
import os
import sys
import torch
import numpy as np
import pickle
import random
import pandas as pd
from datetime import datetime
from dateutil import tz
from torch.nn import CrossEntropyLoss

# import config

from utils import Logger, seed_worker
from train import train_model
from eval import evaluate
from model import Model
from loss import CCCLoss
from dataset import VerBioDataset
from data_parse_sn import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def parse_args():
    parser = argparse.ArgumentParser(description='VerBIO Stress')

    parser.add_argument('--task', type=str, required=True, choices=['wilder', 'sent', 'physio', 'stress'],
                        help='Specify the task (wilder, sent, physio or stress).')
    parser.add_argument('--label_type', default='Mean',
                        help='Specify the emotion dimension (default: Mean).')
    # parser.add_argument('--normalize', action='store_true',
    #                     help='Specify whether to normalize features (default: False).')
    # parser.add_argument('--norm_opts', type=str, nargs='+', default=['n'],
    #                     help='Specify which features to normalize ("y": yes, "n": no) in the corresponding order to '
    #                          'feature_set (default: n).')
    # parser.add_argument('--win_len', type=int, default=200,
    #                     help='Specify the window length for segmentation (default: 200 frames).')
    # parser.add_argument('--hop_len', type=int, default=100,
    #                     help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--d_rnn', type=int, default=64,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=1,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Specify the batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Specify the initial random seed (default: 42).')
    # parser.add_argument('--n_seeds', type=int, default=5,
    #                     help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--save', action='store_true',
                        help='Specify whether to save the best model (default: False).')
    parser.add_argument('--save_path', type=str, default='preds',
                        help='Specify path where to save the predictions (default: preds).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved (default: False).')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')

    parser.add_argument('--label_folder', type=str, default='Baseline',
                        help='Specify label folder for Mean/Selective/Baseline (default: Baseline).')

    args = parser.parse_args()
    return args


def main(args):
    # ensure reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading data ...')
    data_dict = load_data(args.label_type, args.label_folder)
    data_loader = {}
    for partition in data_dict.keys():  # one DataLoader for each partition
        dt = VerBioDataset(data_dict, partition)
        batch_size = args.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False  # shuffle only for train partition
        data_loader[partition] = torch.utils.data.DataLoader(dt, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                                                             worker_init_fn=seed_worker)

    # args.d_in = data_loader['train'].dataset.get_feature_dim()
    args.d_in = 88
    
    if args.task == 'sent':
        args.n_targets = 11  # number of classes
        criterion = CrossEntropyLoss()
        score_str = 'Macro-F1'
    else:
        args.n_targets = 1
        criterion = CCCLoss()
        score_str = 'CCC'

    if args.eval_model is None:  # Train and validate for each seed
        seeds = range(args.seed,args.seed+1)
        val_losses, val_scores, best_model_files, test_scores = [], [], [], []

        for seed in seeds:
            torch.manual_seed(seed)

            model = Model(args)

            print('=' * 50)
            print('Training model... [seed {}]'.format(seed))
            
            val_loss, val_score, best_model_file, res_dict = train_model(args.task, model, data_loader, args.epochs,
                                                               args.lr, seed, args.use_gpu,
                                                               criterion, regularization=args.regularization)

            # if not args.predict:  # run evaluation only if test labels are available
            #     test_loss, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
            #     test_scores.append(test_score)
            #     if args.task in ['physio', 'stress', 'wilder']:
            #         print(f'[Test CCC]:  {test_score:7.4f}')
            val_losses.append(val_loss)
            val_scores.append(val_score)
            best_model_files.append(best_model_file)
            
            with open(f'./logs/{args.log_file_name}.pickle', 'wb') as handle:
                pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            
        # best_idx = val_scores.index(max(val_scores))  # find best performing seed

        # print('=' * 50)
        # print(f'Best {score_str} on [Val] for seed {seeds[best_idx]}: '
        #       f'[Val {score_str}]: {val_scores[best_idx]:7.4f}'
        #       f"{f' | [Test {score_str}]: {test_scores[best_idx]:7.4f}' if not args.predict else ''}")
        # print('=' * 50)

        # model_file = best_model_files[best_idx]  # best model of all of the seeds

    else:  # Evaluate existing model (No training)
        model_file = args.eval_model
        model = torch.load(model_file)
        _, valid_score = evaluate(args.task, model, data_loader['devel'], criterion, args.use_gpu)
        print(f'Evaluating {model_file}:')
        print(f'[Val {score_str}]: {valid_score:7.4f}')
        if not args.predict:
            _, test_score = evaluate(args.task, model, data_loader['test'], criterion, args.use_gpu)
            print(f'[Test {score_str}]: {test_score:7.4f}')

    if args.predict:  # Make predictions for the test partition; this option is set if there are no test labels
        print('Predicting test samples...')
        best_model = torch.load(model_file)
        evaluate(args.task, best_model, data_loader['test'], criterion, args.use_gpu, predict=True,
                 prediction_path=args.paths['predict'])

    # if args.save:  # Save predictions for all partitions (needed to subsequently do late fusion)
    #     print('Save all predictions...')
    #     seed = int(model_file.split('_')[-1].split('.')[0])
    #     torch.manual_seed(seed)
    #     best_model = torch.load(model_file)
    #     # Load data again without any segmentation
    #     data = load_data(args.task, args.paths, args.feature_set, args.emo_dim, args.normalize, args.norm_opts,
    #                      args.win_len, args.hop_len, save=args.cache, apply_segmentation=False)
    #     for partition in data.keys():
    #         dl = torch.utils.data.DataLoader(MuSeDataset(data, partition), batch_size=1, shuffle=False,
    #                                          worker_init_fn=seed_worker)
    #         evaluate(args.task, best_model, dl, criterion, args.use_gpu, predict=True,
    #                  prediction_path=args.paths['save'])

    # Delete model if save option is not set.
    # if not args.save and not args.eval_model:
    #     if os.path.exists(model_file):
    #         os.remove(model_file)

    print('Done.')


if __name__ == '__main__':
    args = parse_args()

    args.log_file_name = '{}_[{}]_[{}]_[{}_{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), '_', args.label_type,
        args.d_rnn, args.rnn_n_layers, args.rnn_bi, args.lr, args.batch_size)

    # adjust your paths in config.py
    # args.paths = {'log': os.path.join(config.LOG_FOLDER, args.task),
    #               'data': os.path.join(config.DATA_FOLDER, args.task),
    #               'model': os.path.join(config.MODEL_FOLDER, args.task, args.log_file_name)}
    # if args.predict:
    #     args.paths['predict'] = os.path.join(config.PREDICTION_FOLDER, args.task, args.log_file_name)
    # if args.save:
    #     args.paths['save'] = os.path.join(args.save_path, args.task, args.log_file_name)
    # for folder in args.paths.values():
    #     if not os.path.exists(folder):
    #         os.makedirs(folder, exist_ok=True)

    # args.paths.update({'features': config.PATH_TO_ALIGNED_FEATURES[args.task],
    #                    'labels': config.PATH_TO_LABELS[args.task],
    #                    'partition': config.PARTITION_FILES[args.task]})

    sys.stdout = Logger(f'./logs/{args.log_file_name}.txt')
    print(' '.join(sys.argv))

    main(args)

    pickle_fid = f'./logs/{args.log_file_name}.pickle'

    with open(pickle_fid, 'rb') as f:
        dt = pickle.load(f)
    
    ccc_list = dt['CCC']
    val_res = {'Label Type': [args.label_type], 'CCC': [np.max(ccc_list)], 'seed': [args.seed]}

    val_res_df = pd.DataFrame(val_res)
 

    val_res_df.to_csv('check_seed_sn2.csv', mode='a', index=False, header=False)



