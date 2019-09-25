# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:58:34 2018

@author: admin
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
import gcn.gnnio as gnnio
import gcn.preprocess as preprocess

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

'''adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels=\
load_data('nell.0.001',1)'''
def load_data(dataset_str,parameters,standard_split=False):
    
    if ('nell' in dataset_str) or (standard_split==True):
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
    
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        idx_all=range(allx.shape[0])
        '''return graph
        print(len(graph))
        for i in tuple(objects):
            print(i.shape)'''
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        '''print(test_idx_range,len(test_idx_range))'''
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
            
            
        if 'nell' in dataset_str:
            # Find relation nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(allx.shape[0], len(graph))#所有relation nodes的id
            isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)#不在reorder的relation nodes的id
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-allx.shape[0], :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-allx.shape[0], :] = ty
            ty = ty_extended
    
            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            #idx_all: all entities
            idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)#包含了reorder的relation nodes（含label的nodes）
    
            if not os.path.isfile("data/planetoid/{}.features.npz".format(dataset_str)):
                print("Creating feature vectors for relations - this might take a while...")
                features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                              dtype=np.float32).tolil()
                #print(len(isolated_node_idx))
                #features_extended[isolated_node_idx, features.shape[1]:] = sp.eye(len(isolated_node_idx))
                datas=features_extended.data
                rows=features_extended.rows
                
                
                for i in range(len(isolated_node_idx)):
                    data_num=1.
                    row=isolated_node_idx[i]
                    col=features.shape[1]+i
                    rows[row]
                    insert_index=0
                    for insert_index in range(len(rows[row])):
                        if rows[row][insert_index]<col:
                            continue
                        elif rows[row][insert_index]>col:
                            break
                    rows[row].insert(insert_index,col)
                    datas[row].insert(insert_index,data_num)
                #features_extended=sp.lil_matrix((datas,rows)) 
                features = sp.csr_matrix(features_extended)
                print("Done!")
                save_sparse_csr("data/planetoid/{}.features".format(dataset_str), features)
            else:
                features = load_sparse_csr("data/planetoid/{}.features.npz".format(dataset_str))
    
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
            
            labels = np.vstack((ally, ty))
            item_number,class_number=labels.shape
            labels[test_idx_reorder, :] = labels[test_idx_range, :]
            label_dict={}
            for row in range(len(labels)):
                for col in range(class_number):
                    if labels[row,col] ==1:
                        if col not in label_dict:
                            label_dict[col]=[]
                        else:
                            label_dict[col].append(row)
            
            if standard_split==True:
                idx_test = test_idx_range.tolist()
                idx_train=[]
                parameters['train_size']=1 
                
                for j in label_dict:
                    idx_train.extend([int(x) for x in list(np.random.choice(label_dict[j],size=parameters['train_size'],replace=False))])
                idx_val=list(np.random.choice(list(set(list(idx_all))-set(idx_train)),size=500,replace=False))
            else:#is nell, not standard split
                testsize=len(idx_test)
                idx_train=[]
                for j in label_dict:
                    idx_train.extend([int(x) for x in list(np.random.choice(label_dict[j],size=parameters['train_size'],replace=False))])
                idx_val=list(np.random.choice(list(set(list(idx_all))-set(idx_train)),size=500,replace=False))
                idx_test=list(np.random.choice(list(set(list(idx_all))-set(idx_train)),size=testsize,replace=False))
        
                
                
        #非nell，是standard split
        else:
            #print(allx.shape)
            features = sp.vstack((allx, tx)).tolil()
            #print(features.shape)
            features[test_idx_reorder, :] = features[test_idx_range, :]
            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]
            
            idx_test = test_idx_range.tolist()
            idx_train = range(parameters['train_size'])
            idx_val = range(parameters['train_size'], parameters['train_size']+500)
        
        
        
    
        
    #not standard split
    
    else:
        data_name='./data/'+dataset_str+'.npz'
        data_graph=gnnio.load_npz_to_sparse_graph(data_name)
        data_graph.adj_matrix = preprocess.eliminate_self_loops(data_graph.adj_matrix)
        data_graph =data_graph.to_undirected()
        data_graph =data_graph.to_unweighted()
        
        '''if dataset_str == 'cora_full':
            # cora_full has some classes that have very few instances. We have to remove these in order for
            # split generation not to fail
            data_graph = preprocess.remove_underrepresented_classes(data_graph,
                                                            parameters['train_size'], 30)
            data_graph= data_graph.standardize()'''
        
        
        labels=data_graph.labels
        adj=data_graph.adj_matrix
        features=data_graph.attr_matrix
        
        try:
            class_number=len(data_graph.class_names)
        except:
            class_number=max(labels)-min(labels)+1
        #convert labels into onehot
        labels_onehot=[]
        for i in labels:
            temp_vector=[]
            for j in range(class_number):
                if j==i:
                    temp_vector.append(1)
                else:
                    temp_vector.append(0)
            labels_onehot.append(temp_vector)
        labels=np.array(labels_onehot)
        
        item_number=labels.shape[0]
        label_dict={}
        for row in range(item_number):
            for col in range(class_number):
                if labels[row,col] ==1:
                    if col not in label_dict:
                        label_dict[col]=[]
                    else:
                        label_dict[col].append(row)
        if parameters['split_seed']:
            np.random.seed(parameters['split_seed'])
        idx_train=[]
        for j in label_dict:
            if len(label_dict[j])>=parameters['train_size']:
                idx_train.extend([int(x) for x in list(np.random.choice(label_dict[j],size=parameters['train_size'],replace=False))])
            else:
                idx_train.extend([int(x) for x in label_dict[j]])
        
        print(idx_train[:20])
        
        idx_val=[]
        '''for j in label_dict:
            remain=list(set(label_dict[j])-set(idx_train))
            if len(remain)>=30:
                idx_val.extend([int(x) for x in list(np.random.choice(remain,size=30,replace=False))])
            else:
                idx_val.extend([int(x) for x in remain])
        idx_val=list(np.random.choice(list(set(list(range(item_number)))-set(idx_train)),size=500,replace=False))'''
        idx_test=list(set(list(range(item_number)))-set(idx_train)-set(idx_val))
        #no need for validation
        
        

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,labels


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask,thresh, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['thresh']: thresh})
    return feed_dict


def chebyshev_polynomials(adj, k):
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def countable(a):
    """
    ----------------------------------------------------------------------
    verify whether an object can be express as numbers
    ----------------------------------------------------------------------
    """
    try:
        b=a+1
        return True
    except:
        return False
    

def str_dict(dictionary,key_name=None,depth=0):
    """
    ----------------------------------------------------------------------
    convert the dict to a reader-friendly string
    ----------------------------------------------------------------------
    """
    if type(dictionary) is dict:
        str_temp=''
        if depth!=0:
            str_temp+='\t'*depth+str(key_name)+':\n'
        for key in dictionary.keys():
            str_temp+=str_dict(dictionary[key],key_name=key,depth=depth+1)
        return str_temp
    else:
        return '{}{}: {}\n'.format('\t'*depth,str(key_name),str(dictionary))
    


class statistic_recorder():
    """
    ----------------------------------------------------------------------
    record the informations and statistics of experiments
    ----------------------------------------------------------------------
    """
    def __init__(self):
        
        self.data={}
        self.statistic={}
    def __str__(self):
        return str_dict(self.statistic)
        
        
        
    def insert(self,thing):
        #'thing' must be a dict with countable values
        for key in thing:
            if countable(thing[key]):
                if key not in self.data:
                    self.data[key]=[]
                self.data[key].append(thing[key])    
        
        
    def update(self):
        
        for key in self.data:
            self.statistic[key]={}
            self.statistic[key]['Times']=len(self.data[key])
            self.statistic[key]['Sum']=sum(self.data[key])
            self.statistic[key]['Mean']=self.statistic[key]['Sum']/self.statistic[key]['Times']
            self.statistic[key]['Deviation']=np.sqrt(sum([(x-self.statistic[key]['Mean'])**2 for x in 
                          self.data[key]]))
        
        return self.statistic
        
from matplotlib import pyplot as plt
def visualize_exps(visualize_parameters,results_list):
    """
    ----------------------------------------------------------------------
    visualize the training porcess
    ----------------------------------------------------------------------
    """
    color_list=['aqua','purple','orchid','coral','beige']
    if 'acc_loss_process' in visualize_parameters:
        for i in range(len(results_list)):
            plt.plot(results_list[i]['recorder']['acc_test'],color=color_list[i],
                     label=results_list[i]['parameters']['thresh'])
            plt.ylim(0.2,1)
        plt.title('Accuracies in training')
        plt.legend()
        plt.savefig('./figure/acc_{}_{}_{}.png'.format(results_list[0]['parameters']['dataset'],
                    results_list[0]['parameters']['train_size'],
                 results_list[0]['parameters']['split_seed']))
        #plt.show()
        
        plt.cla()
        
        '''for results in results_list:
            plt.plot(results['recorder']['loss_test'],color='r')
        plt.title('Losses in training')
        plt.savefig('./figure/loss_{}_{}_{}_{}.png'.format(results['parameters']['dataset'],results['parameters']['train_size'],
                 results['parameters']['split_seed'],results['parameters']['thresh']))
        #plt.show()
        plt.cla()'''
    
    
    
    
    
    
    
    
    
    