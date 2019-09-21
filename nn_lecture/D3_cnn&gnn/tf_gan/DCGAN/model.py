import tensorflow as tf

def batch_norm(x, train_phase, reuse, name='bn_layer'):
    #with tf.variable_scope(name) as scope:
    batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training = train_phase,
            reuse=reuse,
            name=name
    )
    return batch_norm


final_depth = 32
img_size = 32

def conv2d(inputs, depth, reuse, name="conv"):
    Layer = tf.layers.conv2d(inputs,
                            depth,
                            [5,5],
                            strides=(2,2),
                            padding ='same',
                            kernel_initializer=tf.glorot_normal_initializer(),
                            reuse=reuse,
                            name=name)
    return Layer

def conv2d_transpose(inputs, depth, name = "conv_trnaspose"):
    Layer = tf.layers.conv2d_transpose(inputs,
                                       filters=depth,
                                       kernel_size=(5,5),
                                       strides=(2,2),
                                       padding='same',
                                       kernel_initializer=tf.glorot_normal_initializer(),
                                       name=name)
    return Layer


def generator(z, name="generator"):
    with tf.variable_scope("generator"):        
        #inputs = tf.concat(values=[z, y], axis=1)
        inputs = z        
        # project and reshape
        h0 = tf.layers.dense(inputs,
                             int(img_size/16)*int(img_size/16)*final_depth*8,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             name="project")
        h0_reshape = tf.reshape(h0, [-1,int(img_size/16),int(img_size/16),final_depth*8])       
        h0_BA = tf.nn.relu(batch_norm(h0_reshape, True, None, name = "bn0"))        
        print(name," h0_BA ",h0_BA)
        
        h1 = conv2d_transpose(h0_BA,final_depth*4, name = "h1")
        h1_BA = tf.nn.relu(batch_norm(h1, True, None, name = "bn1"))
        print(name," h1_BA ",h1_BA)
        
        h2 = conv2d_transpose(h1_BA,final_depth*2, name = "h2")
        h2_BA = tf.nn.relu(batch_norm(h2, True,None, name = "bn2"))
        print(name," h2_BA ",h2_BA)
        
        h3 = conv2d_transpose(h2_BA,final_depth, name = "h3")
        h3_BA = tf.nn.relu(batch_norm(h3, True,None, name = "bn3"))        
        print(name," h3_BA ",h3_BA)
        
        h4 = conv2d_transpose(h3_BA,1, name = "h4")
        h4_A = tf.nn.tanh(h4)
        print(name," h4_A ",h4_A)      
        return h4_A

def discriminator(x, reuse=None, name="discriminator"):
    with tf.variable_scope("discriminator"):
        batch_size = 64
        x_resize = tf.image.resize_images(x, [img_size,img_size],method=tf.image.ResizeMethod.BILINEAR,align_corners=False)
        #y_map = tf.tile(tf.reshape(y, [batch_size,1,1,10]), [1,img_size,img_size,1])
        
        #inputs = tf.concat(values=[x_resize, y_map], axis=3)
        inputs = x_resize
        
        h0 = conv2d(inputs, final_depth, reuse=reuse, name="h0")
        h0_A = tf.nn.relu(h0)
        print(name," h0_A ",h0_A)
        
        h1 = conv2d(h0_A, final_depth*2, reuse=reuse, name="h1")
        h1_BA = tf.nn.relu(batch_norm(h1, True, reuse, name = "bn1"))
        print(name," h1_BA ",h1_BA)
        
        h2 = conv2d(h1_BA, final_depth*4, reuse=reuse, name="h2")
        h2_BA = tf.nn.relu(batch_norm(h2, True, reuse, name = "bn2"))
        print(name," h2_BA ",h2_BA)
        
        h3 = conv2d(h2_BA, final_depth*8, reuse=reuse, name="h3")
        h3_BA = batch_norm(h3, True, reuse, name = "bn3")
        print(name," h3_BA ",h3_BA)
        
        h4 = tf.layers.dense(tf.reshape(h3_BA, [batch_size, 2*2*final_depth*8]),
                             1,
                             kernel_initializer=tf.glorot_normal_initializer(),
                             reuse=reuse,
                             name="h4")
        print(name," h4 ",h4)
        
        return tf.nn.sigmoid(h4), h4

def loss(D_logit, D, D_logit_, D_):
    # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
    # G_loss = -tf.reduce_mean(tf.log(D_fake))
    
    # Alternative losses:
    # -------------------
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_, labels=tf.zeros_like(D_)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_, labels=tf.ones_like(D_)))
    
    return G_loss, D_loss