###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################

import ConfigParser
import sys

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam

# from keras.utils import plot_model
sys.path.insert(0, './lib/')
from help_functions import *

# function to obtain data for training/testing (validation)
from extract_patches import get_data_training

K.set_image_data_format('channels_first')

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(n_ch, patch_height, patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=1)([
        Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5),
        conv4
    ])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=1)([
        Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6),
        conv3
    ])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=1)([
        Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),
        conv2
    ])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=1)([
        Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8),
        conv1
    ])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(
        optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


# ========= Load settings from Config file
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')
# patch to the datasets
path_data = config.get('data paths', 'path_local')
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths',
                                                     'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths',
                                                  'train_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)

# ========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(
    group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
    './{}/sample_input_imgs'.format(name_experiment))  # .show()
visualize(
    group_images(patches_masks_train[0:N_sample, :, :, :], 5),
    './{}/sample_input_masks'.format(name_experiment))  # .show()

# =========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_unet(n_ch, patch_height, patch_width)  # the U-net model
print "Check: final output of the network:"
print model.output_shape
# plot_model(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./{}/{}_architecture.json'.format(name_experiment, name_experiment),
     'w').write(json_string)

# ============  Training ==================================
checkpointer = ModelCheckpoint(
    filepath='./{}/{}_best_weights.h5'.format(name_experiment, name_experiment),
    verbose=1,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)  # save at each epoch if the validation decreased

# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

# patches_masks_train = masks_Unet(
#     patches_masks_train)  # reduce memory consumption
model.fit(
    patches_imgs_train,
    patches_masks_train,
    epochs=N_epochs,
    batch_size=batch_size,
    verbose=2,
    shuffle=True,
    validation_split=0.1,
    callbacks=[checkpointer])

# ========== Save and test the last model ===================
model.save_weights(
    './{}/{}_last_weights.h5'.format(name_experiment, name_experiment),
    overwrite=True)
