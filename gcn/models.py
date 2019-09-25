from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def label_to_onehot_vec(label,dimension):
    #label is tensor
    batch_size = tf.size(label)
    labels = tf.cast(tf.expand_dims(label, 1),tf.int32)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels],1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, dimension]), 1.0, 0.0)
    return onehot_labels



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
        self.opt_op_binary = self.optimizer.minimize(self.relaxedLoss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self,parameters, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.parameters=parameters
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        #todo
        self.softmaxOutput=tf.nn.softmax(self.outputs)
        self.predictedLabel=tf.argmax(self.outputs,1)
        
        self.onehot_labels=label_to_onehot_vec(self.predictedLabel,dimension=FLAGS.classNumber)
        self.onehot_labels=tf.stop_gradient(self.onehot_labels)
                        
        
        self.classSum=tf.reduce_sum(self.onehot_labels,axis=0)+tf.ones([1,FLAGS.classNumber])
        
        
        
        if FLAGS.attentionFunction=='threshMax':
            """
            ----------------------------------------------------------------------
            Dynamic Self-training Module
            ----------------------------------------------------------------------
            """
            if not self.parameters['linear_weight']:
                self.attention=\
                tf.stop_gradient(
                    tf.multiply(
                        tf.reduce_max(
                            tf.divide(
                                tf.multiply(
                                    (
                                        tf.ones_like(self.softmaxOutput))
                                    ,
                                    tf.cast(
                                        tf.greater(
                                            self.softmaxOutput
                                            ,
                                            self.placeholders['thresh'])
                                        ,
                                        tf.float32))
                                ,
                                self.classSum)
                            ,
                            axis=1)
                        ,
                        (
                            tf.ones_like(
                                self.placeholders['labels_mask']
                                ,
                                dtype=tf.float32)
                            -
                            tf.cast(
                                self.placeholders['labels_mask']
                                ,
                                tf.float32))))
            else:
                self.attention=\
                tf.stop_gradient(
                    tf.multiply(
                        tf.reduce_max(
                            tf.divide(
                                tf.nn.relu(
                                    (                                   
                                        self.softmaxOutput
                                        -
                                        self.placeholders['thresh'])
                                    )
                                ,
                                self.classSum)
                            ,
                            axis=1)
                        ,
                        (
                            tf.ones_like(
                                self.placeholders['labels_mask']
                                ,
                                dtype=tf.float32)
                            -
                            tf.cast(
                                self.placeholders['labels_mask']
                                ,
                                tf.float32))))
                                       
                            
                                    
                                
                
                
                
        elif FLAGS.attentionFunction== 'entropy':
            self.classEqual=tf.reduce_max(tf.divide(self.onehot_labels,self.classSum),axis=1)
            self.attention=tf.stop_gradient(tf.negative(tf.multiply(tf.multiply(tf.reduce_sum(tf.multiply(
                    self.softmaxOutput,tf.log(self.softmaxOutput)
                    
                    ),axis=1),tf.ones_like(self.placeholders['labels_mask'],dtype=tf.float32)-tf.cast(self.placeholders['labels_mask'],tf.float32))
                ,self.classEqual)))
        
        self.relaxedLoss=self.loss+FLAGS.pseudo_label_rate*tf.reduce_sum(tf.multiply(self.attention,  
                                     tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs,
                                                                             labels=self.onehot_labels)  ))
        '''
        
        self.classSum=tf.matmul(self.softmaxOutput,tf.expand_dims(tf.reduce_sum(self.softmaxOutput,axis=0),1))
        self.relaxedLoss=tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.square(tf.multiply(self.softmaxOutput,tf.log(self.softmaxOutput))),axis=1),self.classSum))
        
        
        
        
        self.relaxedLoss=self.loss-FLAGS.retraining_rate*self.relaxedLoss
        '''
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            pooling=FLAGS.pooling,
                                            logging=self.logging,
                                            name='GCN_layer_1'))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            pooling=FLAGS.pooling,
                                            logging=self.logging,
                                            name='GCN_layer_2'))

    def predict(self):
        return tf.nn.softmax(self.outputs)
