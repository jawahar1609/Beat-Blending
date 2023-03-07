import os, numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, concatenate, BatchNormalization, GlobalMaxPool1D, Activation, Conv1D, Flatten, Flatten, Dense

emb_size = 1
alpha = 0.8

# Triplet loss
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    distance1 = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    distance2 = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    loss = tf.reduce_mean(tf.maximum(distance1 - distance2 + alpha, 0))

    return loss

# Build Model
def get_model(r,c) :

    # hyper-parameters
    n_filters = 64
    filter_width = 3
    dilation_rates = [2**i for i in range(8)] 

    history_seq = Input(shape=(r, c))
    x = history_seq

    skips = []
    count = 0
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate, activation='relu', name="conv1d_dilation_"+str(dilation_rate))(x)
        
        x = BatchNormalization()(x)


    out = Conv1D(32, 16, padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('tanh')(out)
    out = GlobalMaxPool1D()(out)
    out = Flatten()(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(history_seq, out)
    model.compile(loss='mse', optimizer='adam')

    input1 = Input((r,c), name="Anchor Input")
    input2 = Input((r,c), name="Positive Input")
    input3 = Input((r,c), name="Negative Input")

    anchor = model(input1)
    positive = model(input2)
    negative = model(input3)


    concat = concatenate([anchor, positive, negative], axis=1)

    siamese = Model([input1, input2, input3], concat)


    siamese.compile(optimizer='adam', loss=triplet_loss)
    print(model.summary())
    print(siamese.summary())

    return model, siamese


data = np.load(os.getcwd()+'/training_siamese_frames.npz')        
anchor = data['a']
pos = data['b']
neg = data['c']

print("Siamese pairs ", anchor.shape, pos.shape, neg.shape)

# Train the model
_,r,c = anchor.shape
encoder, model = get_model(c,r)

anchor = pos.transpose(0, 2, 1)
pos = pos.transpose(0, 2, 1)
neg = neg.transpose(0, 2, 1)

y = np.ones((len(anchor), 1*3))

# Train the model
model.fit([anchor, pos, neg], y, epochs= 30, batch_size= 16, verbose= 1)
model.predict([anchor, pos, neg])
model.save(os.getcwd() + "/siamese.h5")
encoder.save(os.getcwd() + "/encoder.h5")