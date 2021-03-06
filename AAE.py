# -*- coding: utf-8 -*-
"""
AAE on vibration.
10 potential categories but only 6 are identified and sampled.
Ausing AAE, Adversarial AutoEncoder, express the 6 categories in latent space with room for the hidden categories.
In testing, see how the opend and hidden caetgories are projected to the latest code space. 

Maybe it's better to make encoder and decoder to CNN
"""

"""
modes:
1: Latent regulation. train generator to fool Descriminator with reconstruction constraint.
0: Showing latest model results. InOut, true dist, discriminator, latent dist.
"""

exptitle =  'BN_512-256-256-256-256-256-256_lkeakyrelu01_lr001_kp85_1_flatten_alha02_bs1024_ep400' #experiment title that goes in tensorflow folder name
mode= 1
flg_graph = True # showing graphs or not during the training. Showing graphs significantly slows down the training.
model_folder = '' # name of the model to be restored. white space means most recent.
n_leaves = 6  # number of leaves in the mixed 2D Gaussian
n_epochs_ge = 400 #90*n_leaves # mode 3, generator training epochs
ac_batch_size = 1024  # autoencoder training batch size
lr = 0.001
import numpy as np
blanket_resolution = 10*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
dc_real_batch_size = int(blanket_resolution*blanket_resolution/15) # descriminator training real dist samplling batch size

keep_prob = 0.85 # keep probability of drop out
OoT_zWeight = 0.1 # out of target weight for latent z in generator
n_latent_sample = 1000 # latent code visualization sample
tb_batch_size = 400  # x_inputs batch size for tb
tb_log_step = 200  # tb logging step
dc_contour_res_x = 5 # x to the blanket resolution for descriminator contour plot
myColor = ['black','orange', 'red', 'blue','gray','green','pink','cyan','Purple','lime','magenta']
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 512
n_l2 = 256
n_l3 = 256
n_l4 = 256
n_l5 = 256
n_l6 = 256
n_l7 = 256
z_dim = 2
results_path = './Results/AAE'

import tensorflow as tf
from tensorflow.contrib.layers import dropout
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from numpy import array, exp
from math import cos,sin
from tqdm import tqdm
from six.moves import cPickle as pickle
import keras
from sklearn.preprocessing import OneHotEncoder

# resFalseet graph
tf.reset_default_graph()
False
"""
Opening pickled datasets
"""

pfile = r"./Data/WaveImgDatasets.pickle"
with (open(pfile, "rb")) as openfile:
    while True:
        try:
            WIData = pickle.load(openfile)
        except EOFError:
            break

X_test = WIData["test_datasets"]
Y_test = WIData["test_labels"]
X_train = WIData["train_datasets"]
Y_train = WIData["train_labels"]

"""
Removing 4 categories from training dataset
remove category 3,5,7,10
"""
idxTF = np.in1d(Y_train,[3,5,7,10])
Y_train = np.delete(Y_train,np.where(idxTF)[0])
X_train = np.delete(X_train, np.where(idxTF)[0], axis = 0)

np.unique(Y_train)

"""BN_256-256-256-256-256-256-512_lkeakyrelu01_lr001_kp70_1_flatten_alha02_bs1024_ep1500
one hot-encoding
validation dataset
"""
# one-hot encode the labels
ohenc = OneHotEncoder()
ohenc.fit(Y_train.reshape(-1,1))
Y_train_OH = ohenc.transform(Y_train.reshape(-1,1)).toarray()

Y_test_OH = keras.utils.to_categorical(Y_test-1, 10)

# break training set into training and validation sets
(X_train, X_valid) = X_train[1000:], X_train[:1000]
(Y_train, Y_valid) = Y_train_OH[1000:], Y_train_OH[:1000]
Y_test = Y_test_OH

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 45*45*2], name='Input')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='Real_distribution')
real_lbl = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Real_lable')
fake_lbl = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Fake_lable')
unif_z = tf.placeholder(dtype=tf.float32, shape=[blanket_resolution*blanket_resolution, z_dim], name='Uniform_z')
unif_d = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Uniform_digits')
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

"""
Util Functions
"""
def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard10, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}".format(datetime.now().strftime("%Y%m%d%H%M%S"), mode,exptitle)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.makedirs(results_path + folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(log_path)
    return tensorboard_path, saved_model_path, log_path

def get_blanket(resolution):
    resolution = resolution
    xlist = np.linspace(xLU[0], xLU[1], resolution,dtype="float32")
    ylist = np.linspace(yLU[0], yLU[1], resolution,dtype="float32")
    blanket = np.empty((resolution*resolution,2), dtype="float32")
    for i in range(resolution):
        for j in range(resolution):
            blanket[i*resolution+j]=[xlist[j],ylist[i]]
    return xlist,ylist,blanket

def model_restore(saver,pmode,mname=''):
    if pmode == -1 or pmode == 0: # running all or show results -> get the specified model or ese latest one
        if len(mname) > 0:
            all_results = [mname]
        else:
            all_results = [path for path in os.listdir(results_path)] 
    else: # get previous mode
        all_results = [path for path in os.listdir(results_path) if '_'+str(pmode-1)+'_' in path or '_-1_' in path] 
    all_results.sort()
    print(results_path + '/' + all_results[-1] + '/Saved_models/')
    saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
          
"""
Vis Functionsx_input
"""
def show_inout(sess,op, ch):
    """
    Shows input MNIST image and reconstracted image.
    Randomly select 10 images from training dataset.
    Parameters. seess:TF session. op: autoencoder operation
    No return. Displays image.alse
    """
    if not flg_graph:
        return
    idx = random.sample(range(0,len(Y_train)),10)
    img_in = X_train[idx,:,:,:]
    img_out = sess.run(op, feed_dict={x_input: img_in.reshape(-1,45*45*2),is_training:False})
    #.reshape(10,28,28)
    plt.rc('figure', figsize=(15, 3))
    plt.tight_layout()
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(img_in[i][:,:,ch])
        plt.axis('off')
        plt.subplot(2,10,10+i+1)
        plt.imshow(img_out[i][:,:,ch])
        plt.axis('off')
    
    plt.suptitle("Original(1st row) and Decoded(2nd row)")
    plt.show()
    plt.close()

def show_latent_code(sess,spc, ch):
    """
    Shows latent codes distribution based on all MNIST training images, with lables as color.
    Parameters. seess:TF session.
                spc: sample per class
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(8, 8))
    plt.tight_layout()
    
    with tf.variable_scope("Encoder"):
        train_zs = sess.run(encoder_outputZ,feed_dict={x_input:X_train.reshape(-1,45*45*2),is_training:True})

    ytrain = Y_train
    
    cm = matplotlib.colors.ListedColormap(myColor[1:])
    
    fig, ax = plt.subplots(1)
    
    for i in range(6):
        y=train_zs[np.where(ytrain[:,i]==1),1][0,0:spc]
        x=train_zs[np.where(ytrain[:,i]==1),0][0,0:spc]
        color = cm(i)
        ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.02, s = 10)
    
    ax.legend(loc='center left', markerscale = 3, bbox_to_anchor=(1, 0.5))
    ax.set_title('2D latent code')    
    plt.show()
    plt.close()
    

def show_discriminator(sess):
    """X_train
    Shows discriminator activation contour plot. Close to 1 means estimated as positive (true dist).
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    br = blanket_resolution
    xlist, ylist, blanket = get_blanket(br)

    plt.rc('figure', figsize=(6, 5))
    plt.tight_layout()
    
    X, Y = np.meshgrid(xlist, ylist)    
 
    with tf.variable_scope("DiscriminatorZ"):
        desc_result = sess.run(dz_blanket,feed_dict={unif_z:blanket,is_training:True})

    desc_result = 1 / (1 + exp(-desc_result))
    
    Z = np.empty((br,br), dtype="float32")    
    for i in range(br):
        for j in range(br):
            Z[i][j]=desc_result[i*br+j]

    fig, ax = plt.subplots(1)
    cp = ax.contourf(X,Y,Z)
    plt.colorbar(cp)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Descriminator Contour')    
    plt.show()   
    plt.close()
    
def show_real_dist(z_real_dist, real_lbl_ins):
    """
    Shows real distribution
    Parameters. z_real_dist:(batch_size,2) numpy array
    No return. Displays image.
    """
    if not flg_graph:
        return
    plt.rc('figure', figsize=(5, 5))
    plt.Falsetight_layout()
    fig, ax = plt.subplots(1)
    cm = matplotlib.colors.ListedColormap(myColor)

    for i in range(-1,10):
        y=z_real_dist[np.where(real_lbl_ins[:,i]==1),1]
        x=z_real_dist[np.where(real_lbl_ins[:,i]==1),0]
        color = cm(i+1)
        ax.scatter(x,y,label=str(i), alpha=0.9, facecolor=color, linewidth=0.15,s = 5)

    ax.legend(loc='center left',  markerscale = 5, bbox_to_anchor=(1, 0.5))
    ax.set_title('Real Distribution')
    
    plt.xlim(xLU[0],xLU[1])
    plt.ylim(yLU[0],yLU[1])
    plt.show()
    plt.close()

"""
model Functions
"""

def mlp_enc(x): # multi layer perceptron

    alpha = 0.01

    l1 = tf.layers.dense(x, n_l1)
    l1 = tf.layers.batch_normalization(l1, training=is_training)
    l1 = tf.maximum(alpha * l1, l1)
    l1 = dropout(l1, keep_prob, is_training=is_training)
    #        bn1 = tf.contrib.layers.batch_norm(elu1, is_training = is_training)
    #        bn1 = 

    l2 = tf.layers.dense(l1, n_l2)
    l2 = tf.layers.batch_normalization(l2, training=is_training)
    l2 = tf.maximum(alpha * l2, l2)
    l2 = dropout(l2, keep_prob, is_training=is_training)
    
    l3 = tf.layers.dense(l2, n_l3)
    l3 = tf.layers.batch_normalization(l3, training=is_training)
    l3 = tf.maximum(alpha * l3, l3)
    l3 = dropout(l3, keep_prob, is_training=is_training)
    
    l4 = tf.layers.dense(l3, n_l4)
    l4 = tf.layers.batch_normalization(l4, training=is_training)
    l4 = tf.maximum(alpha * l4, l4)
    l4 = dropout(l4, keep_prob, is_training=is_training)
    
    l5 = tf.layers.dense(l4, n_l5)
    l5 = tf.layers.batch_normalization(l4, training=is_training)
    l5 = tf.maximum(alpha * l5, l5)
    l5 = dropout(l5, keep_prob, is_training=is_training)
    
    l6 = tf.layers.dense(l5, n_l6)
    l6 = tf.layers.batch_normalization(l6, training=is_training)
    l6 = tf.maximum(alpha * l6, l6)
    l6 = dropout(l6, keep_prob, is_training=is_training)
    #        elu4 = fully_connected(bn3, n_l4,activation_fn =None)
    #        bn4 = tf.contrib.layers.batch_norm(elu4, is_training = is_training)
    #        bn4 = tf.maximum(alpha * bn4, bn4)
    #        bn4 = dropout(bn4, keep_prob, is_training=is_training)
    return l6

def mlp_dec(x): # multi layer perceptron

    alpha = 0.01
    l1 = tf.layers.dense(x, n_l6)
    l1 = tf.layers.batch_normalization(l1, training=is_training)
    l1 = tf.maximum(alpha * l1, l1)
    l1 = dropout(l1, keep_prob, is_training=is_training)
    
    l2 = tf.layers.dense(l1, n_l5)
    l2 = tf.layers.batch_normalization(l2, training=is_training)
    l2 = tf.maximum(alpha * l2, l2)
    l2 = dropout(l2, keep_prob, is_training=is_training)
    
    l3 = tf.layers.dense(l2, n_l4)
    l3 = tf.layers.batch_normalization(l3, training=is_training)
    l3 = tf.maximum(alpha * l3, l3)
    l3 = dropout(l3, keep_prob, is_training=is_training)
    
    l4 = tf.layers.dense(l3, n_l3)
    l4 = tf.layers.batch_normalization(l4, training=is_training)
    l4 = tf.maximum(alpha * l4, l4)
    l4 = dropout(l4, keep_prob, is_training=is_training)

    l5 = tf.layers.dense(l4, n_l2)
    l5 = tf.layers.batch_normalization(l5, training=is_training)
    l5 = tf.maximum(alpha * l5, l5)
    l5 = dropout(l5, keep_prob, is_training=is_training)
    
    l6 = tf.layers.dense(l5, n_l1)
    l6 = tf.layers.batch_normalization(l6, training=is_training)
    l6 = tf.maximum(alpha * l6, l6)
    l6 = dropout(l6, keep_prob, is_training=is_training)
#        elu1 = fully_connected(bn3, n_l1,activation_fn =None)
#        bn4 = tf.contrib.layers.batch_norm(elu1, is_training = is_training)
#        bn4 = tf.maximum(alpha * bn4, bn4)
#        bn4 = dropout(bn4, keep_prob, is_training=is_training)
    return l6


def encoder(x, reuse=False):
    """
    Encoder part of the autoencoder.
    :param x: input to the autoencoder10
    :param reuse: True -> Reuse the encoder variables, False -> Create the variables
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        last_layer = mlp_enc(x)
        outputZ = tf.layers.dense(last_layer, z_dim,activation=None)
 
    return outputZ

def decoder(z, reuse=False):
    """
    Decoder part of the autoex_inputncoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create the variables
    :return: tensor which should ideally be the input given to the encoder.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        last_layer = mlp_dec(z)
        output = tf.layers.dense(last_layer, 45*45*2, activation=None)
    return output

def discriminator(x, reuse=False):
    """
    Discriminator that leanes to activate at true categorical distribution and not for the others.
    For training, feed the same pair of x and lbl, so it will learn to activate say 3 for 3.
    For generator training, we expect discriminatorY will activate for proper x indicated by the label
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        last_layer = mlp_dec(x)
        output = tf.layers.dense(last_layer, 1,activation=None)
    return output

def gaussian_mixture(batchsize, num_leaves, sel):
    """
    Crate true distribution with num_leaves 2D Gaussian
    :batch_size: number of data points to generate
    :sel: selector. one-hot encoding with last column means unkown category
    :return: tensor of shape [batch_size, 2]. I think it's better to take sigmoid here.
    """
    def sample(x, y, label, num_leaves):
        shift = 1.7
        r = 2.0 * np.pi / float(num_leaves) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.07
    x = np.random.normal(0, x_var, (batchsize, 2 // 2))
    y = np.random.normal(0, y_var, (batchsize, 2 // 2))
    z = np.empty((batchsize, 2), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(2 // 2):
            s = np.random.randint(0, num_leaves) if sel[batch][10] == 1 else np.where(sel[batch]==1)[0][0]
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], s, num_leaves)
    return z

def standardNormal2D(batchsize):
    """
    standard normal 2d dist
    """
    x_var = 1
    y_var = 1
    x = np.random.normal(0, x_var, (batchsize, 1))
    y = np.random.normal(0, y_var, (batchsize, 1))
    z = np.append(x,y,1)
    return z
 
"""
Defining key operations, Loess, Optimizer and other necessary operations
"""
with tf.variable_scope('Encoder'):
    encoder_outputZ = encoder(x_input)

with tf.variable_scope('Decoder'):
    decoder_output = decoder(encoder_outputZ)

with tf.variable_scope('DiscriminatorZ'):
    dz_real = discriminator(real_distribution)
    dz_blanket = discriminator(unif_z, reuse=True)
    dz_fake = discriminator(encoder_outputZ, reuse=True)
    
# loss
with tf.name_scope("ae_loss"):
    autoencoder_loss = tf.reduce_mean(tf.square(x_input - decoder_output))
    autoencoder_loss_v = tf.reduce_mean(tf.square(x_input - decoder_output))
    
with tf.name_scope("dc_loss"):
    dc_zloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dz_real), logits=dz_real))
    dc_zloss_blanket = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dz_blanket), logits=dz_blanket))
    dc_zloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dz_fake), logits=dz_fake))
    dc_zloss = dc_zloss_blanket + dc_zloss_real+dc_zloss_fake

with tf.name_scope("ge_loss"):
    #Out of Target penalty
    OoT_penaltyZ =OoT_zWeight*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dz_fake), logits=dz_fake))
    generator_loss = autoencoder_loss+OoT_penaltyZ #not sure why it averages out

#optimizer
all_variables = tf.trainable_variables()
dc_zvar = [var for var in all_variables if 'DiscriminatorZ/' in var.name]
ae_var = [var for var in all_variables if ('Encoder/' in var.name or 'Decoder/' in var.name)]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope("AE_optimizer"):
        autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(autoencoder_loss,var_list=ae_var)
    
with tf.name_scope("DC_optimizer"):
    discriminatorZ_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(dc_zloss, var_list=dc_zvar)

with tf.name_scope("GE_optimizer"):
    generator_optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(generator_loss, var_list=ae_var)

init = tf.global_variables_initializer()

# Reshape immages to display them
input_images = tf.reshape(x_input, [-1, 45, 45, 2])
generated_images = tf.reshape(decoder_output, [-1, 45, 45, 2])

# Tensorboard visualizationdegit_v
ae_sm = tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)
aev_sm = tf.summary.scalar(name='Autoencoder_LossVal', tensor=autoencoder_loss_v)
dcz_sm = tf.summary.scalar(name='Discriminator_ZLoss', tensor=dc_zloss)
ge_sm = tf.summary.scalar(name='Generator_Loss', tensor=generator_loss)
ootz_sm = tf.summary.scalar(name='OoT_penaltyZ', tensor=OoT_penaltyZ)
summary_op = tf.summary.merge_all()

# Creating saver and get ready
saver = tf.train.Saver()
step = 0

"""
Executing with a session based on mode specification
"""

def tb_init(sess): # create tb path, model path and return tb writer and saved model path
    tensorboard_path, saved_model_path, log_path = form_results()
    sess.run(init)
    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)  
    return writer, saved_model_path

def batch(batch_size):
    perm = np.random.permutation(range(0, len(Y_train)))
    X_train_cpy = X_train[perm]
    Y_train_cpy = Y_train[perm]
    for batch_i in range(0, len(Y_train)//batch_size):
        start_i = batch_i * batch_size
        X_batch = X_train_cpy[start_i:start_i + batch_size].reshape(-1,45*45*2)
        Y_batch = Y_train_cpy[start_i:start_i + batch_size]
        yield X_batch, Y_batch
      
def tb_write(sess, batch_x, batch_y):
    #reuse the others
    aesm = sess.run(ae_sm,feed_dict={x_input: batch_x, is_training:False})
    aevsm = sess.run(aev_sm,feed_dict={x_input: X_valid.reshape(-1,45*45*2), is_training:False})
    gesm = sess.run(ge_sm,feed_dict={x_input: X_valid.reshape(-1,45*45*2), real_distribution:dc_real_dist, 
                                     is_training:False})
    writer.add_summary(aesm, global_step=step)
    writer.add_summary(aevsm, global_step=step)
    writer.add_summary(gesm, global_step=step)


with tf.Session() as sess:
    if mode==1: # Latent regulation
        writer,saved_model_path = tb_init(sess)   
        _,_,blanket = get_blanket(blanket_resolution)
        n_batches = int(len(Y_train) / ac_batch_size)
        for i in range(n_epochs_ge):
            print("------------------Epoch {}/{} ------------------".format(i, n_epochs_ge))
            bt = batch(ac_batch_size)
            for b in tqdm(range(n_batches)):    
                #Discriminator
                batch_x, batch_y = next(bt)
                # real batch uniform sampling for each lable and unknown label. This is not constrained by lable availability.
                dc_real_lbl = np.eye(6)[np.array(np.random.randint(0,6, size=dc_real_batch_size)).reshape(-1)]+np.random.normal(0,0.5)
                dc_real_dist = standardNormal2D(dc_real_batch_size)# or maybe we can make this only smaller
                sess.run([discriminatorZ_optimizer],feed_dict={x_input: batch_x, real_distribution:dc_real_dist,\
                         is_training:True,real_lbl:dc_real_lbl ,unif_z:blanket})
                
                #Generator
                sess.run([generator_optimizer],feed_dict={x_input: batch_x,fake_lbl:batch_y,is_training:True\
                         ,real_distribution:dc_real_dist, unif_z:blanket})
                if b % tb_log_step == 0:
                    show_discriminator(sess) #shows others like 3, 7 -1 ?
                    show_latent_code(sess,n_latent_sample,0)
                    tb_write(sess, batch_x, batch_y)
                step += 1
        saver.save(sess, save_path=saved_model_path, global_step=step, write_meta_graph = True)
        writer.close()
    if mode==0: # showing the latest model result. InOut, true dist, discriminator, latent dist.
        model_restore(saver,mode,model_folder)
        show_inout(sess, op=generated_images, ch=1)       
        #dc_real_dist = standardNormal2D(500)
        #show_discriminator(sess)    
        #show_latent_code(sess,n_latent_sample)
        
    
        