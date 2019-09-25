import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def generator(z, y):
    with tf.variable_scope("generator"):
        
        inputs = tf.concat(values=[z, y],axis=1)
        G_h1 = tf.layers.dense(inputs,
                               128,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_normal_initializer())
        G_prob = tf.layers.dense(G_h1, 784, activation = tf.nn.sigmoid, kernel_initializer=tf.glorot_normal_initializer())
    
        return G_prob

def discriminator(x, y, reuse):
    with tf.variable_scope("discriminator"):       
        inputs = tf.concat(values=[x, y],axis=1)
        D_h1 = tf.layers.dense(inputs,
                               128,
                               reuse = reuse,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_normal_initializer(), 
                               name = "L1")
        D_logit = tf.layers.dense(D_h1, 1, reuse = reuse, kernel_initializer=tf.glorot_normal_initializer(), name = "L2")
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

def loss(D_logit_real, D_logit_fake):
    # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    # G_loss = -tf.reduce_mean(tf.log(D_fake))
    
    # Alternative losses:
    # -------------------
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    
    return G_loss, D_loss

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig