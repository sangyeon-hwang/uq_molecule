import random

import numpy as np
import pandas as pd
from rdkit import Chem

def shuffle_two_list(list1, list2):
    list_total = list(zip(list1,list2))
    random.shuffle(list_total)
    list1, list2 = zip(*list_total)
    return list1, list2

def prepare_data(path1, path2=None, max_atoms=None,
                 test_ratio=0.2, eval_ratio=0.1,
                 initial_shuffle=True):
    """Prepare SMILES and label lists from data files.

    Arguments
    ---------
    path1: str
        Total or training data path.
        If it's a training set, `path2` should be also given.
    path2: str | None
        Test data path.
        If this is None, `path1` is used as a total set.
    test_ratio, eval_ratio: float
        See `split_train_eval_test`.
    initial_shuffle: bool
        Shuffle the data after fetched.
        If `path2` is given, the `path1` data is shuffled
        before splitted into training and validation sets.

    Returns
    -------
    smi_train
    smi_eval
    smi_test
    prop_train
    prop_eval
    prop_test
    """
    # `path1` is the total dataset.
    if path2 is None:
        smi_total, prop_total = load_input(path1, max_atoms)
        if initial_shuffle:
            smi_total, prop_total = shuffle_two_list(smi_total,
                                                     prop_total)
        smi_train, smi_eval, smi_test = split_train_eval_test(
            smi_total, test_ratio, eval_ratio)
        prop_train, prop_eval, prop_test = split_train_eval_test(
            prop_total, test_ratio, eval_ratio)

    # `path1` is the training set and `path2` the test sets.
    else:
        smi_train, prop_train = load_input(path1, max_atoms)
        smi_test, prop_test = load_input(path2, max_atoms)
        if initial_shuffle:
            smi_train, prop_train = shuffle_two_list(smi_train,
                                                     prop_train)
            smi_train, smi_eval, _ = split_train_eval_test(smi_train,
                                                           0.,
                                                           eval_ratio)
            prop_train, prop_eval, _ = split_train_eval_test(prop_train,
                                                             0.,
                                                             eval_ratio)
    return (smi_train, smi_eval, smi_test,
            prop_train, prop_eval, prop_test)

def load_input(path, max_num_atoms=None):
    data = pd.read_csv(path, names=['smiles', 'label'])
    # Filter by number of atoms if requested.
    if max_num_atoms is not None:
        mols = (Chem.MolFromSmiles(smiles) for smiles in data.smiles)
        mask = [mol.GetNumAtoms() <= max_num_atoms if mol else False
                for mol in mols]
        smi_list = data.smiles.loc[mask].tolist()
    else:
        smi_list = data.smiles.tolist()
    prop_list = data.label.tolist()
    return smi_list, prop_list

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
