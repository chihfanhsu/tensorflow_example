import tensorflow as tf

def generator(z, y):
    with tf.variable_scope("generator"):
        
        inputs = tf.concat(values=[z, y],axis=1)
        G_h1 = tf.layers.dense(z,
                               128,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.glorot_normal_initializer())
        G_prob = tf.layers.dense(G_h1, 784, activation = tf.nn.sigmoid, kernel_initializer=tf.glorot_normal_initializer())
    
        return G_prob

def discriminator(x, y, reuse):
    with tf.variable_scope("discriminator"):
        
        inputs = tf.concat(values=[x, y],axis=1)
        D_h1 = tf.layers.dense(x,
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