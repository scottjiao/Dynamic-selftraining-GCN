from __future__ import division
from __future__ import print_function

import os
"""
----------------------------------------------------------------------
Switch the working directory to the file directory
----------------------------------------------------------------------
"""
path=os.path.abspath(os.path.dirname(__file__))
os.chdir(path)


from gcn.emailme import *
import time
import tensorflow as tf
import numpy as np
from gcn.utils_onetime import *
from gcn.models import GCN, MLP


def del_all_flags(FLAGS):
    """
    ----------------------------------------------------------------------
    Delete all configuration information stored in memory.
    ----------------------------------------------------------------------
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
def constructDefaultParameterDict():
    """
    ----------------------------------------------------------------------
    Establish default configuration and hyperparameter settings
    ----------------------------------------------------------------------
    """
    parameters={}
    
    #model hyper-parameters
    thresh=0.9
    epoch=600
    weight_decay=1e-3
    learning_rate=0.01
    hidden1=64
    dropout=0.8
    feature_normalize=False
    linear_weight=False
    retrain=False
    attentionFunction='threshMax'
    pseudo_label_rate=1
    pooling=False
    early_stopping=epoch
    ground_truth=False
    interval_len=5
    addedEpoch=200
    
    
    
    #experiment setups
    dataset='cora'
    print_train=False
    train_size=3
    seedUsing=False
    standardsplit=False
    split_seed=False
    parameter_seed=2018
    #experiment 
    
    parameters['print_train']=print_train
    parameters['attentionFunction']=attentionFunction
    parameters['thresh']=thresh
    parameters['pseudo_label_rate']=pseudo_label_rate
    parameters['seedUsing']=seedUsing
    parameters['train_size']=train_size
    parameters['dataset']=dataset
    parameters['epoch']=epoch
    parameters['retrain']=retrain
    parameters['addedEpoch']=addedEpoch
    parameters['standardsplit']=standardsplit
    parameters['weight_decay']=weight_decay
    parameters['pooling']=pooling
    parameters['learning_rate']=learning_rate
    parameters['hidden1']=hidden1
    parameters['dropout']=dropout
    parameters['early_stopping']=early_stopping
    parameters['interval_len']=interval_len
    parameters['ground_truth']=ground_truth
    parameters['feature_normalize']=feature_normalize
    parameters['linear_weight']=linear_weight
    parameters['split_seed']=split_seed
    parameters['parameter_seed']=parameter_seed
    return parameters

def solid_acc_strategy(inputParameters={}):
    """
    ----------------------------------------------------------------------
    Model main file, input the parameters or configuration collected 
    in the form of a dictionary, and output the result of this training.
    ----------------------------------------------------------------------
    """
    tf.reset_default_graph()
    del_all_flags(tf.flags.FLAGS)
    parameters=constructDefaultParameterDict()
    parameters.update(inputParameters)
    
    # Set random seed
    if parameters['parameter_seed']:
        seed = parameters['parameter_seed']
        #np.random.seed(seed)
        tf.set_random_seed(seed)
    else:
        pass
    
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', parameters['dataset'], 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', parameters['learning_rate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', parameters['epoch'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', parameters['hidden1'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', parameters['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', parameters['weight_decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', parameters['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    #flags.DEFINE_float('thresh', parameters['thresh'], 'thresh.')
    flags.DEFINE_float('pseudo_label_rate', parameters['pseudo_label_rate'], 'pseudo_label_rate.')
    flags.DEFINE_string('attentionFunction', parameters['attentionFunction'], 'attentionFunction.')#threshMax,entropy
    flags.DEFINE_bool('pooling', parameters['pooling'], 'pooling.')
        
    '''
    # =============================================================================
    # #exp settings
    # =============================================================================
    '''
    #pernum=50
    
    # =============================================================================
    #     #load data
    # =============================================================================
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask ,labels= \
            load_data(FLAGS.dataset,parameters,standard_split=parameters['standardsplit'])
    itemNumber,classNumber=labels.shape
    _,featureNumber=features.shape
    flags.DEFINE_integer('itemNumber', itemNumber, 'itemNumber')
    flags.DEFINE_integer('classNumber', classNumber, 'classNumber')
    flags.DEFINE_integer('featureNumber', featureNumber, 'featureNumber')
    
    
        
    """
    ----------------------------------------------------------------------
    preprocess data
    ----------------------------------------------------------------------
    """
    if parameters['feature_normalize']:
        features = preprocess_features(features)
    else:
        features = sparse_to_tuple(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    """
    ----------------------------------------------------------------------
    construct the placeholders
    ----------------------------------------------------------------------
    """
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'thresh': tf.placeholder(tf.float32)  
    }
    
    """
    ----------------------------------------------------------------------
    construct the models and placeholders
    ----------------------------------------------------------------------
    """
    model = model_func(parameters,placeholders, input_dim=features[2][1] ,logging=True)
    sess = tf.Session()
    
    def evaluate(features, support, labels, mask,thresh, placeholders):
        """
        ----------------------------------------------------------------------
        evaluate
        ----------------------------------------------------------------------
        """
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask,thresh, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
    
    
    """
    ----------------------------------------------------------------------
    initialize
    ----------------------------------------------------------------------
    """
    sess.run(tf.global_variables_initializer())
    
    cost_val=[]
    
    acc_train=0
    
    result_recorder={}
    result_recorder['acc_train']=[]
    result_recorder['acc_valid']=[]
    result_recorder['acc_test']=[]
    
    result_recorder['loss_train']=[]
    result_recorder['loss_valid']=[]
    result_recorder['loss_test']=[]
    
    """
    ----------------------------------------------------------------------
    start training
    ----------------------------------------------------------------------
    """
    for epoch in range(FLAGS.epochs):
    
        t = time.time()
        """
        ----------------------------------------------------------------------
        prepare the data to be fed 
        ----------------------------------------------------------------------
        """
        feed_dict = construct_feed_dict(features, support, y_train, train_mask,parameters['thresh'], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        
        
        """
        ----------------------------------------------------------------------
        Adopt the dynamic training process
        ----------------------------------------------------------------------
        """    
        if np.random.rand(1)>acc_train and parameters['ground_truth']:
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        else:
            #min_thresh=1/classNumber
            
            outs = sess.run([model.opt_op_binary, model.relaxedLoss, model.accuracy], feed_dict=feed_dict)
        
        """
        ----------------------------------------------------------------------
        evaluate the model
        ----------------------------------------------------------------------
        """
        cost, acc, duration = evaluate(features, support, y_val, val_mask,1, placeholders)
        cost_val.append(cost)
        
        acc_train=outs[2]
        """
        ----------------------------------------------------------------------
        early stopping
        ----------------------------------------------------------------------
        """
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
        
        """
        ----------------------------------------------------------------------
        evaluate in the test set
        ----------------------------------------------------------------------
        """
        test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask,1, placeholders)
        if parameters['print_train']:
            print("Epoch: {} Test set results:".format(epoch),
              "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "time=", "{:.5f}".format(test_duration))
            
        result_recorder['acc_train'].append(acc_train)
        result_recorder['acc_valid'].append(acc)
        result_recorder['acc_test'].append(test_acc)
        result_recorder['loss_train'].append(outs[1])
        result_recorder['loss_valid'].append(cost)
        result_recorder['loss_test'].append(test_cost)
    
    #print("Optimization Finished!")
    
    # Testing
    test_cost1, test_acc1, test_duration1 = evaluate(features, support, y_test, test_mask,1, placeholders)

    
    return {'enhancedTestAccuracy':test_acc1,'parameters':parameters,'recorder':result_recorder}

