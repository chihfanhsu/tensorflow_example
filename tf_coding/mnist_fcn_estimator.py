
# coding: utf-8

# # Load libraries

# In[1]:


import numpy as np
import tensorflow as tf


# # Load MNIST

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# # Define Model (Estimator)

# In[3]:


def model_fn(features, labels, mode):
    # Define Model parameters
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    # Define Model structure  y = f(x)
    y = tf.matmul(features['x'], W) + b
    
    # Define Model Loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    
    # Define Optimizer and step
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = tf.group(optimizer.minimize(cross_entropy),
                     tf.assign_add(global_step, 1))
    
    # Define other evaluation metrics 
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(tf.argmax(labels,1), tf.argmax(y,1))
    }
    return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=y,
              loss=cross_entropy,
              train_op=train,
              eval_metric_ops=eval_metric_ops)

x_train = mnist.train.images
y_train = mnist.train.labels.astype("float32")
x_eval = mnist.test.images
y_eval = mnist.test.labels.astype("float32")

estimator = tf.estimator.Estimator(model_fn=model_fn)
# define the training & evaluation settings
input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_train},
                y_train,
                batch_size=128,
                num_epochs=None,
                shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_train},
                y_train,
                batch_size=128,
                num_epochs=1,
                shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_eval},
                y_eval,
                batch_size=128,
                num_epochs=1,
                shuffle=False)


# # Training & Evaluation

# In[4]:


# Training
estimator.train(input_fn=input_fn, steps=1000)

# evaluation
train_metrics = estimator.evaluate(input_fn=train_input_fn)["accuracy"]
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)["accuracy"]
print("\nTrain Accuracy: {0:f}\n".format(train_metrics))
print("\nTest Accuracy: {0:f}\n".format(eval_metrics))

