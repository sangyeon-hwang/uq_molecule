import argparse
import os
import random
import sys
import time

import numpy as np
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

import set_path
from models.mc_dropout import mc_dropout
import utils

np.set_printoptions(precision=3)

def predict(model, FLAGS, data):
    total_st = time.time()
    #num_batches = len(smi_list)//FLAGS.batch_size + 1
    Y_pred_total = np.array([])
    Y_batch_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    tot_unc_total = np.array([])
    batches = utils.prepare_batches(data, FLAGS.batch_size)
    for batch in batches:
        A_batch, X_batch = utils.convert_to_graph(batch['smiles'],
                                                  FLAGS.max_atoms)
        if FLAGS.save_truth:
            Y_batch = batch['label']
        
        # MC-sampling
        P_mean = []
        P_logvar = []
        for _ in range(FLAGS.num_samplings):
            Y_mean, Y_logvar = model.predict(A_batch, X_batch)
            P_mean.append(Y_mean.flatten())
            P_logvar.append(Y_logvar.flatten())

        if FLAGS.task_type == 'classification':
            P_mean = utils.np_sigmoid(np.asarray(P_mean))
            ale_unc = np.mean(P_mean*(1.0-P_mean), axis=0)
            epi_unc = np.mean(P_mean**2, axis=0) - np.mean(P_mean, axis=0)**2
        elif FLAGS.task_type == 'regression':
            P_mean = np.asarray(P_mean)
            P_logvar = np.exp(np.asarray(P_logvar))            
            ale_unc = np.mean(P_logvar, axis=0)
            epi_unc = np.var(P_mean, axis=0)
        mean = np.mean(P_mean, axis=0)
        tot_unc = ale_unc + epi_unc

        if FLAGS.save_truth:
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)
        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
        tot_unc_total = np.concatenate((tot_unc_total, tot_unc), axis=0)

    _prefix = FLAGS.output_prefix + '_mc_'
    if FLAGS.save_truth:
        np.save(_prefix + 'truth', Y_batch_total)
    np.save(_prefix + 'pred', Y_pred_total)
    np.save(_prefix + 'epi_unc', epi_unc_total)
    np.save(_prefix + 'ale_unc', ale_unc_total)
    np.save(_prefix + 'tot_unc', tot_unc_total)
    print ("Finish predictions. Total time:", time.time() - total_st)

if __name__ == '__main__':
    # Hyperparameters for a transfer-trained model
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type',
                        default='classification',
                        choices=('classification', 'regression'))
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=32,
                        help='Hidden dimension of graph convolution layers')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=256,
                        help='Hidden dimension of readout & MLP layers')
    parser.add_argument('--max_atoms',
                        type=int,
                        default=150,
                        help='Maximum number of atoms')
    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help='# of hidden layers')
    parser.add_argument('--num_attn',
                        type=int,
                        default=4,
                        help='# of heads for multi-head attention')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size')
    parser.add_argument('--regularization_scale',
                        type=float,
                        default=1e-4,
                        help='')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.9,
                        help='')
    parser.add_argument('--beta2',
                        type=float,
                        default=0.98,
                        help='')
    parser.add_argument('--optimizer',
                        default='Adam',
                        help='Adam | SGD | RMSProp') 
    parser.add_argument('--model_path',
                        help='Name to use in saving models')
    parser.add_argument('--input_path',
                        help='New input path')
    parser.add_argument('--output_prefix',
                        default='PREDICTION',
                        help='Output path prefix')
    parser.add_argument('--num_samplings',
                        type=int,
                        default=20,
                        help='# of MC samplings')
    parser.add_argument('--set_cuda',
                        action='store_true',
                        help='Set CUDA_VISIBLE_DEVICES inside the script')
    FLAGS = parser.parse_args()

    # Load data.
    data = utils.load_input(FLAGS.input_path, FLAGS.max_atoms)
    FLAGS.num_train = len(data)
    # Prepare to save true labels if present.
    if data['label'].notnull().all():
        FLAGS.save_truth = True
    else:
        FLAGS.save_truth = False

    print("Task type:", FLAGS.task_type)
    print("Maximum number of allowed atoms:", FLAGS.max_atoms)
    print("Model path to use:", FLAGS.model_path)
    print("Num data to predict:", len(data))
    print("Num MC samplings:", FLAGS.num_samplings)

    # Set CUDA_VISIBLE_DEVICES if requested.
    if FLAGS.set_cuda:
        utils.set_cuda_devices()
    print("CUDA_VISIBLE_DEVICES=", os.environ.get('CUDA_VISIBLE_DEVICES'))

    model = mc_dropout(FLAGS)
    model.restore(FLAGS.model_path)
    print("Model restored from:", FLAGS.model_path)
    predict(model, FLAGS, data)
