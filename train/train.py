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
from utils import utils

np.set_printoptions(precision=3)

def np_sigmoid(x):
    return 1./(1.+np.exp(-x))

def training(model, FLAGS,
             smi_train, smi_eval, smi_test,
             prop_train, prop_eval, prop_test):
    print ("Start Training XD")
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    init_lr = FLAGS.init_lr
    total_st = time.time()
    #smi_train, smi_eval, smi_test = utils.split_train_eval_test(smi_total, 0.8, 0.2, 0.1)
    #prop_train, prop_eval, prop_test = utils.split_train_eval_test(prop_total, 0.8, 0.2, 0.1)
    #prop_eval = np.asarray(prop_eval)
    #prop_test = np.asarray(prop_test)
    num_train = len(smi_train)
    num_eval = len(smi_eval)
    num_test = len(smi_test)
    #smi_train = smi_train[:num_train]
    #prop_train = prop_train[:num_train]
    num_batches_train = (num_train//batch_size) + 1
    num_batches_eval = (num_eval//batch_size) + 1
    num_batches_test = (num_test//batch_size) + 1
    total_iter = 0
    print("Number of training data:", num_train, "\t evaluation data:", num_eval, "\t test data:", num_test)
    for epoch in range(num_epochs):
        st = time.time()
        lr = init_lr * 0.5**(epoch//10)
        model.assign_lr(lr)
        smi_train, prop_train = utils.shuffle_two_list(smi_train, prop_train)
        prop_train = np.asarray(prop_train)

        # TRAIN
        num = 0
        train_loss = 0.0
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        for i in range(num_batches_train):
            num += 1
            st_i = time.time()
            total_iter += 1
            A_batch, X_batch = utils.convert_to_graph(smi_train[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
            Y_batch = prop_train[i*batch_size:(i+1)*batch_size]

            Y_mean, Y_logvar, loss = model.train(A_batch, X_batch, Y_batch)
            train_loss += loss
            if FLAGS.task_type == 'classification':
                Y_pred = np_sigmoid(Y_mean.flatten())
            elif FLAGS.task_type == 'regression':
                Y_pred = Y_mean.flatten()
            Y_pred_total = np.concatenate((Y_pred_total, Y_pred), axis=0)
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)

            et_i = time.time()

        train_loss /= num
        if FLAGS.task_type == 'classification':
            train_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
            train_auroc = 0.0
            try:
                train_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
            except:
                train_auroc = 0.0    
        elif FLAGS.task_type == 'regression':
            train_mae = np.mean(np.abs(Y_batch_total - Y_pred_total))

        #Eval
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        num = 0
        eval_loss = 0.0
        for i in range(num_batches_eval):
            A_batch, X_batch = utils.convert_to_graph(smi_eval[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
            Y_batch = prop_eval[i*batch_size:(i+1)*batch_size]
        
            # MC-sampling
            P_mean = []
            for n in range(FLAGS.num_eval_samplings):
                num += 1
                Y_mean, Y_logvar, loss = model.test(A_batch, X_batch, Y_batch)
                eval_loss += loss
                P_mean.append(Y_mean.flatten())

            if FLAGS.task_type == 'classification':
                P_mean = np_sigmoid(np.asarray(P_mean))
            elif FLAGS.task_type == 'regression':
                P_mean = np.asarray(P_mean)
            mean = np.mean(P_mean, axis=0)
    
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
            Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)

        eval_loss /= num
        if FLAGS.task_type == 'classification':
            eval_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
            eval_auroc = 0.0
            try:
                eval_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
            except:
                eval_auroc = 0.0    
        elif FLAGS.task_type == 'regression':
            eval_mae = np.mean(np.abs(Y_batch_total - Y_pred_total))

        # Save network! 
        ckpt_path = os.path.join(FLAGS.save_directory,
                                 FLAGS.model_name + '.ckpt')
        model.save(ckpt_path, epoch)
        et = time.time()

        # Print Results
        print ("Time for", epoch, "-th epoch: ", et-st)
        print ("Loss        Train:", round(train_loss,3), "\t Evaluation:", round(eval_loss,3))
        if FLAGS.task_type == 'classification':
            print ("Accuracy    Train:", round(train_accuracy,3), "\t Evaluation:", round(eval_accuracy,3))
            print ("AUROC       Train:", round(train_auroc,3), "\t Evaluation:", round(eval_auroc,3))
        elif FLAGS.task_type == 'regression':
            print ("MAE    Train:", round(train_mae,3), "\t Evaluation:", round(eval_mae,3))
    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))

    #Test
    test_st = time.time()
    Y_pred_total = np.array([])
    Y_batch_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    tot_unc_total = np.array([])
    num = 0
    test_loss = 0.0
    for i in range(num_batches_test):
        num += 1
        A_batch, X_batch = utils.convert_to_graph(smi_test[i*batch_size:(i+1)*batch_size], FLAGS.max_atoms) 
        Y_batch = prop_test[i*batch_size:(i+1)*batch_size]
        
        # MC-sampling
        P_mean = []
        P_logvar = []
        for n in range(FLAGS.num_test_samplings):
            Y_mean, Y_logvar, loss = model.test(A_batch, X_batch, Y_batch)
            P_mean.append(Y_mean.flatten())
            P_logvar.append(Y_logvar.flatten())

        if FLAGS.task_type == 'classification':
            P_mean = np_sigmoid(np.asarray(P_mean))
            ale_unc = np.mean(P_mean*(1.0-P_mean), axis=0)
            epi_unc = np.mean(P_mean**2, axis=0) - np.mean(P_mean, axis=0)**2
        elif FLAGS.task_type == 'regression':
            P_mean = np.asarray(P_mean)
            P_logvar = np.exp(np.asarray(P_logvar))            
            ale_unc = np.mean(P_logvar, axis=0)
            epi_unc = np.var(P_mean, axis=0)
        mean = np.mean(P_mean, axis=0)
        tot_unc = ale_unc + epi_unc
    
        Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)
        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
        tot_unc_total = np.concatenate((tot_unc_total, tot_unc), axis=0)

    _prefix = os.path.join(FLAGS.statistics_directory,
                           FLAGS.model_name + '_mc_')
    np.save(_prefix + 'truth', Y_batch_total)
    np.save(_prefix + 'pred', Y_pred_total)
    np.save(_prefix + 'epi_unc', epi_unc_total)
    np.save(_prefix + 'ale_unc', ale_unc_total)
    np.save(_prefix + 'tot_unc', tot_unc_total)
    test_et = time.time()
    print ("Finish Testing, Total time for test:", (test_et-test_st))

if __name__ == '__main__':
#    dim1 = 32
#    dim2 = 256
#    max_atoms = 150
#    num_layer = 4
#    batch_size = 256
#    epoch_size = 100
#    learning_rate = 0.001
#    regularization_scale = 1e-4
#    beta1 = 0.9
#    beta2 = 0.98

    # Hyperparameters for a transfer-trained model
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type',
                        default='classification',
                        help='classification | regression')
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
    parser.add_argument('--epoch_size',
                        type=int,
                        default=100,
                        help='Epoch size')
#    tf.flags.DEFINE_integer('num_train',
#                            num_train,
#                            'Number of training data')
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
    parser.add_argument('--init_lr',
                        type=float,
                        default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--max_to_keep',
                        type=int,
                        default=5,  # None for unlimited.
                        help='`max_to_keep` for `tf.train.Saver`')
    parser.add_argument('--model_name',
                        default='MC-Dropout',
                        help='Name to use in saving models')
    parser.add_argument('--data_path',
                        default=None,
                        help='Total-data path')
    parser.add_argument('--train_data_path',
                        default=None,
                        help='Training-data path. ALSO NEEDS: test_data_path')
    parser.add_argument('--test_data_path',
                        default=None,
                        help='Test-data path. ALSO NEEDS: train_data_path')
    parser.add_argument('--save_directory',
                        default='save',
                        help='directory path to save models')
    parser.add_argument('--statistics_directory',
                        default='statistics',
                        help='directory path to save statistics results')
    parser.add_argument('--num_eval_samplings',
                        type=int,
                        default=3,
                        help='# of MC samplings in validation')
    parser.add_argument('--num_test_samplings',
                        type=int,
                        default=20,
                        help='# of MC samplings in test time')
    parser.add_argument('--set_cuda',
                        action='store_true',
                        help='Set CUDA_VISIBLE_DEVICES inside the script')
    FLAGS = parser.parse_args()

    # Load data.
#    smi_total, prop_total = load_input_HIV()
#    num_total = len(smi_total)
#    num_test = int(num_total*0.2)
#    num_train = num_total-num_test
#    num_eval = int(num_train*0.1)
#    num_train -= num_eval
    if FLAGS.data_path:
        (smi_train, smi_eval, smi_test,
         prop_train, prop_eval, prop_test) = utils.prepare_data(
            FLAGS.data_path)
    elif FLAGS.train_data_path and FLAGS.test_data_path:
        (smi_train, smi_eval, smi_test,
         prop_train, prop_eval, prop_test) = utils.prepare_data(
            FLAGS.train_data_path, FLAGS.test_data_path)
    else:
        raise RuntimeError(
            "either of 'data_path' or 'train_data_path' "
            "and 'test_data_path' should be given!")
    # Training-data size is used by `models.mc_dropout.mc_dropout`.
    FLAGS.num_train = len(smi_train)

    # Prepare the directories to save results.
    if not os.path.exists(FLAGS.save_directory):
        os.mkdir(FLAGS.save_directory)
    if not os.path.exists(FLAGS.statistics_directory):
        os.mkdir(FLAGS.statistics_directory)

    print("Do Single-Task Learning")
    print("Task type:", FLAGS.task_type)
    print("Hidden dimension of graph convolution layers:", FLAGS.hidden_dim)
    print("Hidden dimension of readout & MLP layers:", FLAGS.latent_dim)
    print("Maximum number of allowed atoms:", FLAGS.max_atoms)
    print("Batch size:", FLAGS.batch_size,
          "\tEpoch size:", FLAGS.epoch_size)
    print("Initial learning rate:", FLAGS.init_lr,
          "\tBeta1:", FLAGS.beta1,
          "\tBeta2:", FLAGS.beta2,
          "\tfor the ", FLAGS.optimizer, " optimizer used in this training")

    # Set CUDA_VISIBLE_DEVICES if requested.
    if FLAGS.set_cuda:
        utils.set_cuda_devices()
    print("CUDA_VISIBLE_DEVICES=", os.environ.get('CUDA_VISIBLE_DEVICES'))

    model = mc_dropout(FLAGS)
    training(model, FLAGS,
             smi_train, smi_eval, smi_test,
             prop_train, prop_eval, prop_test)
