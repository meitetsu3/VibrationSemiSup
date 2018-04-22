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
exptitle =  'base' #experiment title that goes in tensorflow folder name
mode= 1
flg_graph = False # showing graphs or not during the training. Showing graphs significantly slows down the training.
model_folder = '' # name of the model to be restored. white space means most recent.
n_leaves = 6  # number of leaves in the mixed 2D Gaussian
n_epochs_ge = 90*n_leaves # mode 3, generator training epochs
ac_batch_size = 100  # autoencoder training batch size
import numpy as np
blanket_resolution = 10*int(np.sqrt(n_leaves)) # blanket resoliution for descriminator or its contour plot
dc_real_batch_size = int(blanket_resolution*blanket_resolution/15) # descriminator training real dist samplling batch size

OoT_zWeight = 1 # out of target weight for latent z in generator
OoT_yWeight = 1 # out of target weight for latent y in generator
n_latent_sample = 5000 # latent code visualization sample
tb_batch_size = 400  # x_inputs batch size for tb
tb_log_step = 200  # tb logging step
dc_contour_res_x = 5 # x to the blanket resolution for descriminator contour plot
myColor = ['black','orange', 'red', 'blue','gray','green','pink','cyan','Purple','lime','magenta']
xLU = [-10,10] # blanket x axis lower and upper
yLU = [-10,10] # blanket y axis lower and upper
n_l1 = 500
n_l2 = 500
z_dim = 2
results_path = './Results/AAE'

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import random
from math import cos,sin
from tqdm import tqdm
from six.moves import cPickle as pickle
import keras
from sklearn.preprocessing import OneHotEncoder

# reset graph
tf.reset_default_graph()

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

"""
one hot-encoding
validation dataset
"""
# one-hot encode the labels
ohenc = OneHotEncoder()
ohenc.fit(Y_train.reshape(-1,1))
Y_train_OH = ohenc.transform(Y_train.reshape(-1,1)).toarray()
Y_train_OH.shape

Y_test_OH = keras.utils.to_categorical(Y_test-1, 10)

# break training set into training and validation sets
(X_train, X_valid) = X_train[1000:], X_train[:1000]
(Y_train, Y_valid) = Y_train_OH[1000:], Y_train_OH[:1000]
Y_test = Y_test_OH

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 45,45,2], name='Input')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='Real_distribution')
real_lbl = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Real_lable')
fake_lbl = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Fake_lable')
unif_z = tf.placeholder(dtype=tf.float32, shape=[blanket_resolution*blanket_resolution, z_dim], name='Uniform_z')
unif_d = tf.placeholder(dtype=tf.float32, shape=[None,6],name = 'Uniform_digits')
he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

"""
Util Functions
"""
def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
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
    saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
          
"""
Vis Functions
"""
def show_inout(sess,op, ch):
    """
    Shows input MNIST image and reconstracted image.
    Randomly select 10 images from training dataset.
    Parameters. seess:TF session. op: autoencoder operation
    No return. Displays image.
    """
    if not flg_graph:
        return
    idx = random.sample(len(Y_train),ac_batch_size)
    img_in = X_train[idx,:,:,ch]
    img_out = sess.run(op, feed_dict={x_input: img_in})
    img_out_s = img_out.reshape(ac_batch_size,45,45)
    img_in_s = img_in.reshape(ac_batch_size,45,45)
    #.reshape(10,28,28)
    plt.rc('figure', figsize=(15, 3))
    plt.tight_layout()
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(img_in_s[i],cmap="gray")
        plt.axis('off')
        plt.subplot(2,10,10+i+1)
        plt.imshow(img_out_s[i],cmap="gray")
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
        train_zs,train_ys = sess.run(encoder(X_train, reuse=True))
    ytrain = Y_train
    
    cm = matplotlib.colors.ListedColormap(myColor[1:])
    
    fig, ax = plt.subplots(1)
    
    for i in range(10):
        y=train_zs[np.where(ytrain[:,i]==1),1][0,0:spc]
        x=train_zs[np.where(ytrain[:,i]==1),0][0,0:spc]
        color = cm(i)
        ax.scatter(x, y, label=str(i), alpha=0.9, facecolor=color, linewidth=0.02, s = 10)
    
    ax.legend(loc='center left', markerscale = 3, bbox_to_anchor=(1, 0.5))
    ax.set_title('2D latent code')    
    plt.show()
    plt.close()
    
def show_discriminator(sess,digit):
    """
    Shows discriminator activation contour plot. Close to 1 means estimated as positive (true dist).
    Parameters. seess:TF session.
    No return. Displays image.
    """
    if not flg_graph:
        return
    br = dc_contour_res_x*blanket_resolution
    xlist, ylist, blanket = get_blanket(br)

    digit_v = np.zeros((len(blanket),11),dtype=np.float32)
    digit_v[:,digit] = 1.0
    
    plt.rc('figure', figsize=(6, 5))
    plt.tight_layout()
    
    X, Y = np.meshgrid(xlist, ylist)    
    
    with tf.variable_scope("DiscriminatorZ"):
        desc_result = sess.run(tf.nn.sigmoid(discriminator(blanket,digit_v, reuse=True)))

    Z = np.empty((br,br), dtype="float32")    
    for i in range(br):
        for j in range(br):
            Z[i][j]=desc_result[i*br+j]

    fig, ax = plt.subplots(1)
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Descriminator Contour for digit '+ str(digit))    
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
    plt.tight_layout()
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
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        elu1 = fully_connected(x, n_l1,scope='elu1')
        elu2 = fully_connected(elu1, n_l2,scope='elu2')
    return elu2

def mlp_dec(x): # multi layer perceptron
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_initializer=he_init):
        elu2 = fully_connected(x, n_l2,scope='elu2')
        elu1 = fully_connected(elu2, n_l1,scope='elu1')
    return elu1

def encoder(x, reuse=False):
    """
    Encoder part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create the variables
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_enc(x)
    outputZ = fully_connected(last_layer, z_dim,weights_initializer=he_init, scope='linZ',activation_fn=None)
 
    return outputZ

def decoder(z, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create the variables
    :return: tensor which should ideally be the input given to the encoder.
    tf.sigmoid
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(z)
    output = fully_connected(last_layer, 2, weights_initializer=he_init, scope='Sigmoid', activation_fn=tf.sigmoid)
    return output

def discriminator(x, reuse=False):
    """
    Discriminator that leanes to activate at true categorical distribution and not for the others.
    For training, feed the same pair of x and lbl, so it will learn to activate say 3 for 3.
    For generator training, we expect discriminatorY will activate for proper x indicated by the label
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    last_layer = mlp_dec(x)
    output = fully_connected(last_layer, 1, weights_initializer=he_init, scope='None',activation_fn=None)
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

with tf.name_scope("AE_optimizer"):
    autoencoder_optimizer = tf.train.AdamOptimizer().minimize(autoencoder_loss)

with tf.name_scope("DC_optimizer"):
    discriminatorZ_optimizer = tf.train.AdamOptimizer().minimize(dc_zloss, var_list=dc_zvar)

with tf.name_scope("GE_optimizer"):
    generator_optimizer = tf.train.AdamOptimizer().minimize(generator_loss, var_list=ae_var)

init = tf.global_variables_initializer()

# Reshape immages to display them
input_images = tf.reshape(x_input, [-1, 45, 45, 2])
generated_images = tf.reshape(decoder_output, [-1, 45, 45, 2])

# Tensorboard visualizationdegit_v
ae_sm = tf.summary.scalar(name='Autoencoder_Loss', tensor=autoencoder_loss)
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

def next_batch(batch_size):
   for batch_i in range(0, len(Y_train)//batch_size):
      start_i = batch_i * batch_size
      X_batch = X_train[start_i:start_i + batch_size]
      Y_batch = Y_train[start_i:start_i + batch_size]

      return X_batch, Y_batch
      
def tb_write(sess):
    batch_x, batch_y = next_batch(tb_batch_size)
    #reuse the others
    sm = sess.run(summary_op,feed_dict={x_input: batch_x, real_distribution:dc_real_dist,\
             real_lbl:dc_real_lbl ,unif_z:blanket, unif_d:blanket_d, fake_lbl:batch_y})
    writer.add_summary(sm, global_step=step)

with tf.Session() as sess:
    if mode==1: # Latent regulation
        writer,saved_model_path = tb_init(sess)   
        _,_,blanket = get_blanket(blanket_resolution)
        n_batches = int(len(Y_train) / ac_batch_size)
        for i in range(n_epochs_ge):
            print("------------------Epoch {}/{} ------------------".format(i, n_epochs_ge))
            for b in tqdm(range(n_batches)):    
                #Discriminator
                batch_x, batch_y = next_batch(ac_batch_size)
                
                # real batch uniform sampling for each lable and unknown label. This is not constrained by lable availability.
                dc_real_lbl = np.eye(6)[np.array(np.random.randint(0,6, size=dc_real_batch_size)).reshape(-1)]+np.random.normal(0,0.5)
                dc_real_dist = standardNormal2D(dc_real_batch_size)# or maybe we can make this only smaller
                
                blanket_d = np.eye(6)[np.array(np.random.randint(0,6, size=blanket_resolution*blanket_resolution)).reshape(-1)]
                blanket_y = np.random.uniform(-10, 10, (blanket_resolution*blanket_resolution,6))
                sess.run([discriminatorZ_optimizer],feed_dict={x_input: batch_x, real_distribution:dc_real_dist,\
                         real_lbl:dc_real_lbl ,unif_z:blanket, unif_d:blanket_d, fake_lbl:batch_y})
                
                #Generator
                sess.run([generator_optimizer],feed_dict={x_input: batch_x,fake_lbl:batch_y})
                if b % tb_log_step == 0:
                    show_discriminator(sess,1) #shows others like 3, 7 -1 ?
                    show_latent_code(sess,n_latent_sample,0)
                    tb_write(sess)
                step += 1
        saver.save(sess, save_path=saved_model_path, global_step=step, write_meta_graph = True)
        writer.close()
    if mode==0: # showing the latest model result. InOut, true dist, discriminator, latent dist.
        model_restore(saver,mode,model_folder)
        show_inout(sess, op=decoder_output) 
        dc_real_lbl = np.eye(6)[np.array(np.random.randint(0,6, size=5000)).reshape(-1)]        
        dc_real_dist = standardNormal2D(500)
        show_discriminator(sess,0)    
        show_discriminator(sess,5)
        show_latent_code(sess,n_latent_sample)
        
    
        