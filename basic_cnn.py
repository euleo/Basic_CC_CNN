import numpy as np
import glob
import os
import tensorflow as tf
import scipy.io
import math     
import random
import itertools
import PIL
import time
import datetime

from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Lambda, AveragePooling2D
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import SGD
from scipy.ndimage.filters import gaussian_filter 
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import regularizers

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_gt_from_mat(gt_file, gt_shape):
    '''
    @brief: This function creates density map from matlab file with points annotations.
    @param: gt_file: matlab file with annotated points.
    @param: gt_shape: density map shape.
    @return: density map and number of points for the input matlab file.
    '''
    gt = np.zeros(gt_shape, dtype='float32')
    mat_contents = scipy.io.loadmat(gt_file)
    dots = mat_contents['annPoints']
    for dot in dots:
        gt[int(math.floor(dot[1]))-1, int(math.floor(dot[0]))-1] = 1.0
    sigma = 15
    density_map = gaussian_filter(gt, sigma)
    return density_map, len(dots)

def euclideanDistanceCountingLoss(yTrue,yPred):
    counting_loss = K.mean(K.square(yTrue - yPred))
    return counting_loss    
    
def mse(pred, gt):
    return np.sqrt(((pred - gt) ** 2.0).mean())

def mae(pred, gt):
    return abs(pred - gt).mean()
    
def step_decay(epoch):
    initial_lrate = 1e-6
    drop = 0.1
    epochs_drop = int(round((iterations/iterations_per_epoch)/2))
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
    
def downsample(im, size):
    rng_y = np.linspace(0, im.shape[0], size[0]+1).astype('int')
    rng_y = list(zip(rng_y[:-1], rng_y[1:]))
    rng_x = np.linspace(0, im.shape[1], size[1]+1).astype('int')
    rng_x = list(zip(rng_x[:-1], rng_x[1:]))        
    res = np.zeros(size)
    for (yi, yr) in enumerate(rng_y):
        for (xi, xr) in enumerate(rng_x):
            res[yi, xi] = im[yr[0]:yr[1], xr[0]:xr[1]].sum()    
    return res
    
def main():
    counting_dataset_path = 'counting_data_UCF'
    counting_dataset = list()
    train_labels = {}
    val_labels = {}
    for im_path in glob.glob(os.path.join(counting_dataset_path, '*.jpg')):
        counting_dataset.append(im_path)
        img = image.load_img(im_path)
        gt_file = im_path.replace('.jpg','_ann.mat')
        h,w = img.size
        dmap,crowd_number = load_gt_from_mat(gt_file, (w,h))
        train_labels[im_path] = dmap
        val_labels[im_path] = crowd_number
        
    mae_sum = 0.0
    mse_sum = 0.0

    # create folder to save results
    date = str(datetime.datetime.now())
    d = date.split()
    d1 = d[0]
    d2 = d[1].split(':')    
    results_folder = 'Results-'+d1+'-'+d2[0]+'.'+d2[1]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
    
    # 5-fold cross validation
    epochs = int(round(iterations/iterations_per_epoch))
    n_fold = 5
    for f in range(0,n_fold):
        print('\nFold '+str(f))
        
        vgg = VGG16(include_top=False, weights=None, input_shape=(None,None,3))
        transfer_layer = vgg.get_layer('block5_conv3')        
        vgg_partial = Model(inputs=vgg.input, outputs=transfer_layer.output, name='vgg_partial')
                   
        # Start a new Keras Sequential model.
        train_model = Sequential()
        
        # Add the convolutional part of the VGG16 model from above.
        train_model.add(vgg_partial)

        train_model.add(Conv2D(1, (3, 3),strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='counting_output'))
        train_model.summary()  

        # l2 weight decay
        for layer in train_model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularizers.l2(5e-4)        
            elif layer.name == 'vgg_partial':
                for l in layer.layers:                    
                    if hasattr(l, 'kernel_regularizer'):
                        l.kernel_regularizer = regularizers.l2(5e-4)                                  
                
        optimizer = SGD(lr=0.0, decay=0.0, momentum=0.9, nesterov=False)
        loss={'counting_output': euclideanDistanceCountingLoss}
        train_model.compile(optimizer=optimizer,
                        loss=loss)                      

        if f == 0:
            split_train = ['counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}#dizionario con 10 corrispondenze image_path-->crowd_number
            split_train_labels = {k: train_labels[k] for k in split_train}#dizionario con 40 corrispondenze image_path-->dmap
        elif f == 1:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}#dizionario con 10 corrispondenze image_path-->crowd_number
            split_train_labels = {k: train_labels[k] for k in split_train}#dizionario con 40 corrispondenze image_path-->dmap
        elif f == 2:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}#dizionario con 10 corrispondenze image_path-->crowd_number
            split_train_labels = {k: train_labels[k] for k in split_train}#dizionario con 40 corrispondenze image_path-->dmap
        elif f == 3:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val = ['counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}#dizionario con 10 corrispondenze image_path-->crowd_number
            split_train_labels = {k: train_labels[k] for k in split_train}#dizionario con 40 corrispondenze image_path-->dmap
        elif f == 4:
            split_train = ['counting_data_UCF/37.jpg','counting_data_UCF/48.jpg','counting_data_UCF/29.jpg','counting_data_UCF/10.jpg','counting_data_UCF/14.jpg','counting_data_UCF/1.jpg','counting_data_UCF/45.jpg','counting_data_UCF/47.jpg','counting_data_UCF/40.jpg','counting_data_UCF/24.jpg','counting_data_UCF/25.jpg','counting_data_UCF/49.jpg','counting_data_UCF/18.jpg','counting_data_UCF/13.jpg','counting_data_UCF/28.jpg','counting_data_UCF/34.jpg','counting_data_UCF/17.jpg','counting_data_UCF/3.jpg','counting_data_UCF/26.jpg','counting_data_UCF/15.jpg','counting_data_UCF/31.jpg','counting_data_UCF/6.jpg','counting_data_UCF/33.jpg','counting_data_UCF/2.jpg','counting_data_UCF/30.jpg','counting_data_UCF/36.jpg','counting_data_UCF/42.jpg','counting_data_UCF/20.jpg','counting_data_UCF/38.jpg','counting_data_UCF/11.jpg','counting_data_UCF/5.jpg','counting_data_UCF/7.jpg','counting_data_UCF/4.jpg','counting_data_UCF/21.jpg','counting_data_UCF/27.jpg','counting_data_UCF/39.jpg','counting_data_UCF/22.jpg','counting_data_UCF/43.jpg','counting_data_UCF/32.jpg','counting_data_UCF/35.jpg']
            split_val = ['counting_data_UCF/8.jpg','counting_data_UCF/50.jpg','counting_data_UCF/12.jpg','counting_data_UCF/19.jpg','counting_data_UCF/44.jpg','counting_data_UCF/23.jpg','counting_data_UCF/9.jpg','counting_data_UCF/46.jpg','counting_data_UCF/16.jpg','counting_data_UCF/41.jpg']
            split_val_labels = {k: val_labels[k] for k in split_val}#dizionario con 10 corrispondenze image_path-->crowd_number
            split_train_labels = {k: train_labels[k] for k in split_train}#dizionario con 40 corrispondenze image_path-->dmap
        
        X_counting = np.empty((len(split_train), 224, 224, 3))         
        y_counting = np.empty((len(split_train),14,14,1))        
        y_tmp = np.empty((len(split_train),14,14)) # to temporarily save the resized counting target  
        for i, imgpath in enumerate(split_train):
            counting_img = image.load_img(imgpath)           
            crop_resized_img = counting_img.resize((224,224),PIL.Image.BILINEAR)
            crop_resized_array_img = image.img_to_array(crop_resized_img)            
            X_counting[i,] = crop_resized_array_img
            
            dmap = split_train_labels[imgpath]
            y_tmp[i] = downsample(dmap, (14, 14))            
            y_counting[i] = np.resize(y_tmp[i],(14,14,1))                           
                  
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]        
        train_model.fit(x = X_counting, y = y_counting, batch_size = batch_size, epochs=epochs, callbacks=callbacks_list)

        predictions = np.empty((len(split_val),1))
        y_validation = np.empty((len(split_val),1))
        for i in range(len(split_val)):
            img = image.load_img(split_val[i]) # test image original size
#            img = image.load_img(split_val[i], target_size=(224, 224)) # test image 224x224
            img_to_array = image.img_to_array(img)  
            img_to_array = np.expand_dims(img_to_array, axis=0)
        
            pred_test = train_model.predict(img_to_array)
            predictions[i] = np.sum(pred_test)
            y_validation[i] = split_val_labels[split_val[i]]        
        
        mean_abs_err = mae(predictions, y_validation)
        mean_sqr_err = mse(predictions, y_validation)               
        
        # serialize model to JSON
        model_json = train_model.to_json()
        model_json_name = "test_model_"+str(f)+".json"
        with open(model_json_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model_h5_name = "test_model_"+str(f)+".h5"
        train_model.save_weights(model_h5_name)
        print("Saved model to disk")
        
        print('\n######################')
        print('Results on TEST SPLIT:')
        print(' MAE: {}'.format(mean_abs_err))
        print(' MSE: {}'.format(mean_sqr_err))
        print("Took %f seconds" % (time.time() - s))
        path1 = results_folder+'/test_split_results_fold-'+str(f)+'.txt'
        with open(path1, 'w') as f:
            f.write('mae: %f,\nmse: %f, \nTook %f seconds' % (mean_abs_err,mean_sqr_err,time.time() - s))

        mae_sum = mae_sum + mean_abs_err
        mse_sum = mse_sum + mean_sqr_err
    
    print('\n################################')
    print('Average Results on TEST SPLIT:')    
    print(' AVE MAE: {}'.format(mae_sum/n_fold))
    print(' AVE MSE: {}'.format(mse_sum/n_fold))
    print("Took %f seconds" % (time.time() - s))
    path2 = results_folder+'/test_split_results_avg.txt'
    with open(path2, 'w') as f:
        f.write('avg_mae: %f, \navg_mse: %f, \nTook %f seconds' % (mae_sum/n_fold,mse_sum/n_fold,time.time() - s))
        
if __name__ == "__main__":
    s = time.time()
    batch_size = 25
    iterations = 20000
    train_split_length = 40
    iterations_per_epoch = int(round((train_split_length)/batch_size))
    main()

