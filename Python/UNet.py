#
# model UNET
#
#    Творчески взято и переработано: https://github.com/pietz/unet-keras
#
#

from keras.models import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Conv2D, Concatenate, Activation, Dropout
from keras.layers.normalization import BatchNormalization

def level_block(m, dim, depth, acti, dropout=0.25, batch=True):
    if depth > 0:
        
        x = m
        if batch : x = BatchNormalization()(x)
        n = Conv2D(dim, (3, 3), activation=acti, padding='same')(x) #(m)
        n = Conv2D(dim, (3, 3), activation=acti, padding='same')(n)
        m = MaxPooling2D((2, 2))(n)
        if dropout : m = Dropout(dropout)(m) ## add
        m = level_block(m, 2*dim, depth-1, acti, dropout=dropout, batch=batch)
        m = UpSampling2D((2, 2))(m)
        m = Conv2D(dim, (2, 2), activation=acti, padding='same', kernel_initializer='he_normal')(m)
        ### 2017-08-01  !!!!!!!!!!! if dropout : m = Dropout(dropout)(m) ## add
        m = Concatenate(axis=3)([n, m])
    m = Conv2D(dim, (3, 3), activation=acti, padding='same')(m)
    return Conv2D(dim, (3, 3), activation=acti, padding='same')(m)

def UNet(img_shape, n_out=2, dim=32, depth=5, acti='relu', dropout=0.25, batch=True):
    i = Input(shape=img_shape)
    ## ???? i = ZeroPadding2D((6,6),data_format="channels_last")
    o = level_block(i, dim, depth, acti, dropout=dropout, batch=batch)
    o = Conv2D(n_out, (1, 1), name='prediction',activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)

if 0 :
    model10 = UNet((512,512,3),depth=4+1,n_out=1, dropout=0.05, batch=False)
    model10.summary()
