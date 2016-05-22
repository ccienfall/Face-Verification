import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from PIL import Image
import pickle
from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from data import data

num_samples, train_set, test_set, val_set, tr_scheme, tr_stream, val_scheme, val_stream, test_scheme, test_stream = data()

def target_transform(X):
    return floatX(X).transpose(0, 3, 1, 2)/127.5 - 1.

def input_transform(X):
    return target_transform(X)

l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 2            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images
batch_size = 128
nx = npx*npx*nc   # # of dimensions in X
niter = 1000        # # of iter at starting learning rate
niter_decay = 30   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 25000   # # of examples to train on

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

def mse(x,y):
    return T.sum(T.pow(x-y,2), axis = 1)

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
sigma_ifn = inits.Normal(loc = -100., scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

def make_conv_layer(X, input_size, output_size, input_filters, 
                    output_filters, name, index,
                    weights = None, filter_sz = 5):
    
    is_deconv = output_size >= input_size

    w_size = (input_filters, output_filters, filter_sz, filter_sz) \
            if is_deconv else (output_filters, input_filters, filter_sz, filter_sz)
    
    if weights is None:
        w = gifn(w_size, '%sw%i' %(name, index))
        g = gain_ifn((output_filters), '%sg%i' %(name, index))
        b = bias_ifn((output_filters), '%sb%i' %(name, index))
    else:
        w,g,b = weights
    
    conv_method = deconv if is_deconv else dnn_conv
    activation = relu if is_deconv else lrelu
    
    sub = output_size / input_size if is_deconv else input_size / output_size
    
    if filter_sz == 3:
        bm = 1
    else:
        bm = 2
    
    layer = activation(batchnorm(conv_method(X, w, subsample=(sub, sub), border_mode=(bm, bm)), g=g, b=b))
    
    return layer, [w,g,b]

def make_conv_set(input, layer_sizes, num_filters, name, weights = None, filter_szs = None):
    assert(len(layer_sizes) == len(num_filters))
    
    vars_ = []
    layers_ = []
    current_layer = input
    
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        input_filters = num_filters[i]
        output_filters = num_filters[i + 1]
        
        if weights is not None:
            this_wts = weights[i * 3 : i * 3 + 3]
        else:
            this_wts = None
            
        if filter_szs != None:
            filter_sz = filter_szs[i]
        else:
            filter_sz = 5
        
        layer, new_vars = make_conv_layer(current_layer, input_size, output_size, 
                                          input_filters, output_filters, name, i, 
                                          weights = this_wts, filter_sz = filter_sz)
        
        vars_ += new_vars
        layers_ += [layer]
        current_layer = layer
        
    return current_layer, vars_, layers_

# inputs
X = T.tensor4()

## encode layer
e_layer_sizes = [128, 32, 8]
e_filter_sizes = [1, 256, 1024]

eX, e_params, e_layers = make_conv_set(X, e_layer_sizes, e_filter_sizes, "e")

## generative layer
g_layer_sizes = [8, 16, 32, 64, 128]
g_num_filters = [1024, 512, 256, 256, 128]

g_out, g_params, g_layers = make_conv_set(eX, g_layer_sizes, g_num_filters, "g")
gwx = gifn((128, 2, 5, 5), 'gwx')
g_params += [gwx]
gX = tanh(deconv(g_out, gwx, subsample=(1, 1), border_mode=(2, 2)))

## discrim layer(s)

df1 = 128
d_layer_sizes = [128, 64, 32, 16, 8]
d_filter_sizes = [2, df1, 2 * df1, 4 * df1, 8 * df1]

dwy = difn((df1 * 8 * 10 * 8, 1), 'dwy')

def discrim(input, name, weights=None):
    d_out, disc_params, d_layers = make_conv_set(input, d_layer_sizes, d_filter_sizes, name, weights = weights)
    d_flat = T.flatten(d_out, 2)
    
    disc_params += [dwy]
    y = sigmoid(T.dot(d_flat, dwy))
    
    return y, disc_params, d_layers

# target outputs
target = T.tensor4()

p_real, d_params, d_layers = discrim(target, "d")
#we need to make sure the p_gen params are the same as the p_real params
p_gen , d_params2, d_layers = discrim(gX, "d", weights=d_params)

# test everything working so far (errors are most likely size mismatches)
f = theano.function([X], p_gen)
#f(input_transform(x_train)).shape


from theano.tensor.signal.downsample import max_pool_2d

## GAN costs
d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

## MSE encoding cost is done on an (averaged) downscaling of the image
#target_pool = max_pool_2d(target, (4,4), mode="average_exc_pad",ignore_border=True)
target_flat = T.flatten(target, 2)
#gX_pool = max_pool_2d(gX, (4,4), mode="average_exc_pad",ignore_border=True)
gX_flat = T.flatten(gX,2)
enc_cost = mse(gX_flat, target_flat).mean() 

## generator cost is a linear combination of the discrim cost plus the MSE enocding cost
d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d + enc_cost / 100   ## if the enc_cost is weighted too highly it will take a long time to train

## N.B. e_cost and e_updates will only try and minimise MSE loss on the autoencoder (for debugging)
e_cost = enc_cost

cost = [g_cost_d, d_cost_real, enc_cost]

elrt = sharedX(0.002)
lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
e_updater = updates.Adam(lr=elrt, b1=b1, regularizer=updates.Regularizer(l2=l2))

d_updates = d_updater(d_params, d_cost)
g_updates = g_updater(e_params + g_params, g_cost)
e_updates = e_updater(e_params, e_cost)

print 'COMPILING'
t = time()
_train_g = theano.function([X, target], cost, updates=g_updates)
_train_d = theano.function([X, target], cost, updates=d_updates)
_train_e = theano.function([X, target], cost, updates=e_updates)
_get_cost = theano.function([X, target], cost)
print '%.2f seconds to compile theano functions'%(time()-t)

img_dir = "/data/chencj/Face/gen_images/"

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

ae_encode = theano.function([X, target], [gX, target])

def inverse(X):
    X_pred = (X.transpose(0, 2, 3, 1) + 1) * 127.5
    X_pred = np.rint(X_pred).astype(int)
    X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
    return X_pred.astype('uint8')
YR = (np.array([1.164,0,1.596,1.164,-0.392,-0.813,1.164,2.017,0],dtype = float)).reshape(3,3)

def YUV2RGB(im):
    target = np.array(im)
    for i in range(160):
        for j in range(128):
            target[i,j]=(np.dot(YR,(target[i,j]-np.array([16,128,128])).reshape(3,1))).reshape(3)
    return target
def save_sample_pictures():
    for te_train, te_target in test_stream.get_epoch_iterator():
        break
    te_out, te_ta = ae_encode(input_transform(te_train), target_transform(te_target))
    te_reshape = inverse(te_out)
    te_target_reshape = inverse(te_ta)

    new_size = (128 * 6, 160 * 12)
    new_im = Image.new('RGB', new_size)
    r = np.random.choice(128, 24, replace=False).reshape(2,12)
    for i in range(2):
        for j in range(12):
            index =  r[i][j]
            a1 = np.concatenate((te_train[index],te_target_reshape[index]),axis=2)
            a1 = YUV2RGB(a1)
            a2 = np.concatenate((te_train[index],te_train[index]),axis=2)
            a2 = np.concatenate((a2,te_train[index]),axis=2)
            a3 = np.concatenate((te_train[index],te_reshape[index]),axis=2)
            a3 = YUV2RGB(a3)
            target_im = Image.fromarray(a1.astype(np.uint8))
            train_im = Image.fromarray(a2.astype(np.uint8))
            im = Image.fromarray(a3.astype(np.uint8))
            
            new_im.paste(target_im, (128 * i * 3, 160 * j))
            new_im.paste(train_im, (128 * (i * 3 + 1), 160 * j))
            new_im.paste(im, (128 * (i * 3 + 2), 160 * j))
    img_loc = "/data/chencj/Face/gen_images/%i.png" %int(time())     
    print "saving images to %s" %img_loc
    new_im.save(img_loc)
    
save_sample_pictures()

#exit(0)
def mn(l):
    if sum(l) == 0:
        return 0
    return sum(l) / len(l)

## TODO : nicer way of coding these means?

def get_test_errors():
    print "getting test error"
    g_costs = []
    d_costs = []
    e_costs = []
    k_costs = []
    for i in range(20):
        try:
            x_train, x_target  = te_iterator.next()
        except:
            te_iterator = val_stream.get_epoch_iterator()
            x_train, x_target  = te_iterator.next()
        x = input_transform(x_train)
        t = target_transform(x_target)
        cost = _get_cost(x,t)
        g_cost, d_cost, enc_cost = cost
        g_costs.append(g_cost)
        d_costs.append(d_cost)
        e_costs.append(enc_cost)
        
    s= " ,".join(["test errors :", str(mn(g_costs)), str(mn(d_costs)), str(mn(e_costs))])
    return s
iterator = tr_stream.get_epoch_iterator()

# you may wish to reset the learning rate to something of your choosing if you feel it is too high/low
lrt = sharedX(lr)

from time import time

n_updates = 0
t = time()

n_epochs = 200

print "STARTING"



for epoch in range(n_epochs):
    
    
    tm = time()

    g_costs = []
    d_costs = []
    e_costs = []
    
    ## TODO : produces pretty ugly output, redo this?
    for i in tqdm(range(num_samples/128)):
        
        try:
            x_train, x_target  = iterator.next()
        except:
            iterator = tr_stream.get_epoch_iterator()
            x_train, x_target  = iterator.next()
        x = input_transform(x_train)
        t = target_transform(x_target)

        ## optional - change the criteria for how often we train the generator or discriminator
        if n_updates % 2 == 1:
            cost = _train_g(x,t) 
        else:
            cost = _train_d(x,t)
            
        # optional - only train the generator on MSE cost
        #cost = _train_e(x,t)
        g_cost, d_cost, enc_cost = cost
        g_costs.append(g_cost)
        d_costs.append(d_cost)
        e_costs.append(enc_cost)

        if n_updates % 100 == 0:
            s= " ,".join(["training errors :", str(mn(g_costs)), str(mn(d_costs)), str(mn(e_costs))])
            g_costs = []
            d_costs = []
            e_costs = []
            print get_test_errors()
            print s
            sys.stdout.flush()
            save_sample_pictures()
	    #all_params = [e_params, g_params, d_params]

            #pickle.dump(all_params, open("faces_dcgan_%s.pkl"%n_updates, 'w'))
        n_updates += 1  

    print "epoch %i of %i took %.2f seconds" %(epoch, n_epochs, time() - tm)
    
    ## optional - reduce the learning rate as you go
    #lrt.set_value(floatX(lrt.get_value() * 0.95))
    #print lrt.get_value()
    
    
    sys.stdout.flush()
    if epoch % 10 ==0:
        all_params = [e_params,g_params,d_params]
        pickle.dump(all_params,open("/data/chencj/Face/params/faces_dcgan_%s.pkl"%epoch,'w'))

all_params = [e_params, g_params, d_params]

pickle.dump(all_params, open("faces_dcgan.pkl", 'w'))
