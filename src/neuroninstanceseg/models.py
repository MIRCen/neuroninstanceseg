import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout

def unet(patch_height, patch_width, n_ch, act ='selu', category_num = 3):
    inputs = Input((patch_height, patch_width, n_ch))
    conv1 = Conv2D(32, (3, 3), activation=act, padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation=act, padding='same')(conv1)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=act, padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation=act, padding='same')(conv2)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation=act, padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=act, padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation=act, padding='same')(conv4)
    conv4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation=act, padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(256, (3, 3), activation=act, padding='same')(conv5)
    conv5 = Dropout(0.2)(conv5)
    # up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up1 = layers.concatenate([up1, conv4])

    conv6 = Conv2D(256, (3, 3), activation=act, padding='same')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation=act, padding='same')(conv6)
    conv6 = Dropout(0.2)(conv6)
    # up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up2 = layers.concatenate([up2, conv3])

    conv7 = Conv2D(128, (3, 3), activation=act, padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation=act, padding='same')(conv7)
    conv7 = Dropout(0.2)(conv7)
    # up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up3 = layers.concatenate([up3, conv2])

    conv8 = Conv2D(64, (3, 3), activation=act, padding='same')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(32, (3, 3), activation=act, padding='same')(conv8)
    conv8 = Dropout(0.2)(conv8)
    # up4 = UpSampling2D(size=(2, 2))(conv8)
    up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up4 = layers.concatenate([up4, conv1])

    #
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation=act, padding='same')(conv9)
    conv9 = Dropout(0.2)(conv9)

    o = Conv2D(category_num, (1, 1), padding='same', activation="softmax")(conv9)
    model = Model(inputs=inputs, outputs=o)

    return model

def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
        
# @tf.function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    # tf.print('dice_coef => (2. * intersection + 1) : ', (2. * intersection + 1))
    # tf.print('dice_coef => K.sum(y_true_f):', K.sum(y_true_f))
    # tf.print('dice_coef => K.sum(y_true_f):', K.sum(y_pred_f))
    # tf.print('dice_coef => (K.sum(y_true_f) + K.sum(y_pred_f) + 1):', (K.sum(y_true_f) + K.sum(y_pred_f) + 1))
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# @tf.function
def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))

@tf.function
def softmax_dice_loss(y_true, y_pred):
    v1 = categorical_crossentropy(y_true, y_pred) * 0.6
    v2 = dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2
    v3 = dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2
    # tf.print('softmax_dice_loss => v1:', v1)
    # tf.print('softmax_dice_loss => v2:', v2)
    # tf.print('softmax_dice_loss => v3:', v3)
    return v1 + v2 + v3
    # return categorical_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.3

# @tf.function
def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# @tf.function
def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)