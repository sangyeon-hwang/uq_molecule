import os
import random

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import train_test_split

def available_cuda_devices():
    import pynvml
    pynvml.nvmlInit()
    total = pynvml.nvmlDeviceGetCount()
    idle_idxs = []
    for idx in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        if not pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            idle_idxs.append(idx)
    return idle_idxs

def set_cuda_devices(num_devices=1):
    device_ids = [str(idx) for idx in available_cuda_devices()]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids[:num_devices])

def np_sigmoid(x):
    return 1./(1.+np.exp(-x))

def classification_scores(y_truth, y_pred):
    """Compute several binary-classification scores.

    Returns
    -------
    accuracy
    auroc
    precision
    recall
    f1_score
    """
    y_truth = y_truth.astype(int)
    try:
        auroc = metrics.roc_auc_score(y_truth, y_pred)
    except:
        print("Error in computing AUROC!")
        auroc = 0.0
    # Convert predictions to binaries for the other scores.
    y_pred = np.around(y_pred).astype(int)
    accuracy = metrics.accuracy_score(y_truth, y_pred)
    precision = metrics.precision_score(y_truth, y_pred)
    recall = metrics.recall_score(y_truth, y_pred)
    f1_score = metrics.f1_score(y_truth, y_pred)
    return accuracy, auroc, precision, recall, f1_score

def shuffle_two_list(list1, list2):
    list_total = list(zip(list1,list2))
    random.shuffle(list_total)
    list1, list2 = zip(*list_total)
    return list1, list2

def prepare_batches(data, batch_size, pos_neg_ratio=None, sample=False):
    """Batch generator.

    Arguments
    ---------
    data: pd.DataFrame
    batch_size: int
    pos_neg_ratio: List[Union[int, float]] of length 2
        Sampling ratios of positive and negative data points.
        `data['label']` should only contain 0s and 1s.
        If None, samples follow the original population.
    sample: bool
        Whether to prepare batches by random samplings or not.
        Always True if `pos_neg_ratio` is given.

    Returns
    -------
    Generator of `pd.DataFrame`, each of length `batch_size`.
    """
    num_batches = len(data) // batch_size

    # If batch_size > len(data), yield one batch by upsampling.
    if not num_batches:
        yield data.sample(n=batch_size, replace=True)

    for i in range(num_batches):
        if pos_neg_ratio:
            num_pos = data['label'].sum()
            num_neg = len(data) - num_pos
            weights = np.ones(len(data))
            pos_weight = num_neg/num_pos * pos_neg_ratio[0]/pos_neg_ratio[1]
            weights[data['label']==1] *= pos_weight
            yield data.sample(n=batch_size, weights=weights, replace=True)

        elif sample:
            yield data.sample(n=batch_size)

        else:
            yield data.iloc[i*batch_size : (i+1)*batch_size]

def prepare_data(path1, path2=None, max_atoms=None,
                 test_size=0.2, eval_size=0.1,
                 shuffle=True):
    """Prepare SMILES and label data from text files.

    The returned three `pd.DataFrame`s have 'smiles' and 'label'
    for column names.

    Arguments
    ---------
    path1: str
        Total or training data path.
        If it's a training set, `path2` should be also given.
    path2: str | None
        Test data path.
        If this is None, `path1` is used as a total set.
    test_size, eval_size: int | float
        Number or portion of test/eval data.
        If float, should be between 0.0 and 1.0.
    shuffle: bool
        Shuffle the data before split.

    Returns
    -------
    train_data: pd.DataFrame
    eval_data: pd.DataFrame
    test_data: pd.DataFrame
    """
    # `path1` is the total dataset.
    if path2 is None:
        total_data = load_input(path1, max_atoms)
        train_eval, test_data = train_test_split(total_data,
                                                 test_size=test_size,
                                                 shuffle=shuffle)
        train_data, eval_data = train_test_split(train_eval,
                                                 test_size=eval_size,
                                                 shuffle=shuffle)

    # `path1` is the training set and `path2` the test sets.
    else:
        train_eval = load_input(path1, max_atoms)
        train_data, eval_data = train_test_split(train_eval,
                                                 test_size=eval_size,
                                                 shuffle=shuffle)
        test_data = load_input(path2, max_atoms)

    return train_data, eval_data, test_data

def load_input(path, max_atoms=None) -> pd.DataFrame:
    data = pd.read_csv(path, names=['smiles', 'label'])

    # Filter by max number of atoms if requested.
    if max_atoms is None:
        mask = [True] * len(data)
    else:
        mask = (Chem.MolFromSmiles(smiles).GetNumAtoms() <= max_atoms
                for smiles in data['smiles'])

    return data.loc[mask]

def load_input_zinc():
    smi_list = []
    prop_list = []
    f = open('./data/ZINC/zinc_all.txt', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list

def load_input_HIV():    
    smi_list = []
    prop_list = []
    # Actives
    f = open('./data/HIV/HIV_all.txt', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list


def load_input_cep():
    smi_list = []
    prop_list = []
    f = open('./data/CEP/cep-processed.csv', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list

def load_input_dude(target_name):
    smi_list = []
    prop_list = []
    f = open('./data/dude/'+target_name+'_all.txt', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list

def load_input_tox21(tox_name, max_atoms):
    f = open('./data/tox21/'+tox_name+'_all.txt', 'r')
    lines = f.readlines()
    smi_list = []
    prop_list = []
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        else:    
            if( m.GetNumAtoms() < max_atoms+1 ):
                smi_list.append(smi)
                prop_list.append(prop)
    return smi_list, prop_list

def split_train_eval_test(input_list, test_ratio, eval_ratio):
    """Split a total set into training, validation, and test subsets.

    Actual split ratios:
        (train + eval) : test = (1-test_ratio) : test_ratio
        train : eval = (1-eval_ratio) : eval_ratio
    """
    num_total = len(input_list)
    num_test = int(num_total*test_ratio)
    num_train = num_total-num_test
    num_eval = int(num_train*eval_ratio)
    num_train -= num_eval

    train_list = input_list[:num_train]
    eval_list = input_list[num_train:num_train+num_eval]
    test_list = input_list[num_train+num_eval:]
    return train_list, eval_list, test_list

def convert_to_graph(smiles_list, max_atoms):
    adj = []
    features = []
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= max_atoms):
            # Feature-preprocessing
            iFeature = np.zeros((max_atoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((max_atoms, max_atoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), 1))

    features = np.asarray(features)
    adj = np.asarray(adj)
    return adj, features

def adj_k(adj, k):
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)  
    return convert_adj(ret)

def convert_adj(adj):

    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d

def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
