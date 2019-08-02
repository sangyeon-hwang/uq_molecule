import argparse
import os
import random
import sys
import time

import numpy as np
from rdkit import Chem
#from sklearn.metrics import accuracy_score, roc_auc_score
#from sklearn.model_selection import train_test_split
import tensorflow as tf

import set_path
from models.mc_dropout import mc_dropout
from utils import utils

np.set_printoptions(precision=3)

def training(model, FLAGS, train_data, eval_data, test_data):
    print ("Start Training XD")
    total_st = time.time()
    print("Number of training data:", len(train_data), end='')
    print("\t evaluation data:", len(eval_data), end='')
    print("\t test data:", len(test_data))
    for epoch in range(FLAGS.epoch_size):
        st = time.time()
        lr = FLAGS.init_lr * 0.5**(epoch//10)
        model.assign_lr(lr)
        # Shuffle the training data at every epoch.
        train_data = train_data.sample(frac=1)

        # TRAIN
        train_nll = 0.0
        train_loss = 0.0
        train_mse = 0.0
        train_mlv = 0.0
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        train_batches = utils.prepare_batches(train_data,
                                              FLAGS.batch_size, 
                                              FLAGS.pos_neg_ratio)
        for batch_idx, batch in enumerate(train_batches):
            st_i = time.time()
            A_batch, X_batch = utils.convert_to_graph(batch['smiles'],
                                                      FLAGS.max_atoms)
            Y_batch = batch['label']
            Y_mean, Y_logvar, nll, loss, mse, mlv = model.train(A_batch, X_batch, Y_batch)
            train_nll += nll
            train_loss += loss
            train_mse += mse
            train_mlv += mlv

            if FLAGS.task_type == 'classification':
                Y_pred = utils.np_sigmoid(Y_mean.flatten())
            elif FLAGS.task_type == 'regression':
                Y_pred = Y_mean.flatten()
            Y_pred_total = np.concatenate((Y_pred_total, Y_pred), axis=0)
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)

        train_nll /= batch_idx + 1
        train_loss /= batch_idx + 1
        train_mse /= batch_idx + 1
        train_mlv /= batch_idx + 1

        if FLAGS.task_type == 'classification':
            (train_accuracy,
             train_auroc,
             train_precision,
             train_recall,
             train_f1_score) = utils.classification_scores(Y_batch_total,
                                                           Y_pred_total)
        elif FLAGS.task_type == 'regression':
            train_mae = np.mean(np.abs(Y_batch_total - Y_pred_total))

        #Eval
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        eval_nll = 0.0
        eval_loss = 0.0
        eval_mse = 0.0
        eval_mlv = 0.0
        eval_batches = utils.prepare_batches(eval_data, FLAGS.batch_size)
        for batch_idx, batch in enumerate(eval_batches):
            A_batch, X_batch = utils.convert_to_graph(batch['smiles'],
                                                      FLAGS.max_atoms)
            Y_batch = batch['label']
        
            # MC-sampling
            P_mean = []
            for _ in range(FLAGS.num_eval_samplings):
                Y_mean, Y_logvar, nll, loss, mse, mlv = model.test(A_batch, X_batch, Y_batch)
                eval_nll += nll
                eval_loss += loss
                eval_mse += mse
                eval_mlv += mlv
                P_mean.append(Y_mean.flatten())

            if FLAGS.task_type == 'classification':
                P_mean = utils.np_sigmoid(np.asarray(P_mean))
            elif FLAGS.task_type == 'regression':
                P_mean = np.asarray(P_mean)
            mean = np.mean(P_mean, axis=0)
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
            Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)

        eval_nll /= (batch_idx + 1) * FLAGS.num_eval_samplings
        eval_loss /= (batch_idx + 1) * FLAGS.num_eval_samplings
        eval_mse /= (batch_idx + 1) * FLAGS.num_eval_samplings
        eval_mlv /= (batch_idx + 1) * FLAGS.num_eval_samplings

        if FLAGS.task_type == 'classification':
            (eval_accuracy,
             eval_auroc,
             eval_precision,
             eval_recall,
             eval_f1_score) = utils.classification_scores(Y_batch_total,
                                                          Y_pred_total)
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
        print ("NLL         Train:", round(train_nll,3), "\t Evaluation:", round(eval_nll,3))
        if FLAGS.task_type == 'classification':
            print ("Accuracy    Train:", round(train_accuracy,3), "\t Evaluation:", round(eval_accuracy,3))
            print ("AUROC       Train:", round(train_auroc,3), "\t Evaluation:", round(eval_auroc,3))
            print ("Precision   Train:", round(train_precision,3), "\t Evaluation:", round(eval_precision,3))
            print ("Recall      Train:", round(train_recall,3), "\t Evaluation:", round(eval_recall,3))
            print ("F1 score    Train:", round(train_f1_score,3), "\t Evaluation:", round(eval_f1_score,3))
        elif FLAGS.task_type == 'regression':
            print ("MSE         Train:", round(train_mse,3), "\t Evaluation:", round(eval_mse,3))
            print ("MLV         Train:", round(train_mlv,3), "\t Evaluation:", round(eval_mlv,3))
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
    test_loss = 0.0
    test_batches = utils.prepare_batches(test_data, FLAGS.batch_size)
    for batch in test_batches:
        A_batch, X_batch = utils.convert_to_graph(batch['smiles'],
                                                  FLAGS.max_atoms)
        Y_batch = batch['label']
        
        # MC-sampling
        P_mean = []
        P_logvar = []
        for _ in range(FLAGS.num_test_samplings):
            Y_mean, Y_logvar, nll, loss, mse, mlv = model.test(A_batch, X_batch, Y_batch)
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
    parser.add_argument('--epoch_size',
                        type=int,
                        default=100,
                        help='Epoch size')
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
    parser.add_argument('--pos_neg_ratio',
                        nargs=2,
                        type=float,
                        help='positive:negative ratio in training batches')
    FLAGS = parser.parse_args()

    # Load data.
    if FLAGS.data_path:
        train_data, eval_data, test_data = utils.prepare_data(
            FLAGS.data_path,
            max_atoms=FLAGS.max_atoms)
    elif FLAGS.train_data_path and FLAGS.test_data_path:
        train_data, eval_data, test_data = utils.prepare_data(
            FLAGS.train_data_path,
            FLAGS.test_data_path,
            max_atoms=FLAGS.max_atoms)
    else:
        message = ("either of 'data_path', or 'train_data_path' "
                   "and 'test_data_path' should be given!")
        raise RuntimeError(message)

    # Training-data size is used by `models.mc_dropout.mc_dropout`.
    FLAGS.num_train = len(train_data)

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
    training(model, FLAGS, train_data, eval_data, test_data)
