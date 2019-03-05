import numpy as np
import tensorflow as tf

def batch_norm(x, train_phase, name='bn_layer'):
    #with tf.variable_scope(name) as scope:
    batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training = train_phase,
            name=name
    )
    return batch_norm

# def conv_blk (inputs,n_filter, train_phase, name = 'conv_blk'):
#     with tf.variable_scope(name):
#         c1 = tf.layers.conv2d(inputs, filters=n_filter[0], kernel_size=[3,3], strides=(1,1), padding='same')       
#         c1_bn = batch_norm(c1, train_phase, name='c1_bn')
#         c1_relu = tf.nn.relu(c1_bn)
#         c2 = tf.layers.conv2d(c1_relu,filters=n_filter[1],kernel_size=[3,3],strides=(1,1),padding='same')        
#         c2_bn = batch_norm(c2, train_phase, name='c2_bn')
#         c2_relu = tf.nn.relu(c2_bn)
#         return c2_relu
    
def fc_blk (inputs, n_nodes, train_phase, name= 'fc_blk'):
    with tf.variable_scope(name):
        fc = tf.layers.dense(inputs, n_nodes,activation=None)
        fc_bn = batch_norm(fc, train_phase, name='fc_bn')
        fc_relu = tf.nn.relu(fc_bn)
        return fc_relu


def inference(model_input, keep_prob, train_phase):
    
    # encoder
    en_1 = fc_blk(model_input, 4096, train_phase, name='en_1')
    en_2 = fc_blk(en_1, 1024, train_phase, name='en_2')
    en_3 = fc_blk(en_2, 512, train_phase, name='en_3')      
    en_4 = fc_blk(en_3, 256, train_phase, name='en_4')
    
    # decoder
    de_1 = fc_blk(en_4, 512, train_phase, name='de_1') 
    de_2 = fc_blk(de_1, 1024, train_phase, name='de_2')
    de_3 = fc_blk(de_2, 4096, train_phase, name='de_3')
            
    de_4 = tf.layers.dense(de_3, 30*450,activation=None)

    return en_4, de_4