from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Flatten, Dropout, Reshape,RepeatVector,Lambda
import tensorflow as tf
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import History
from keras.layers import Input
from keras import optimizers
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from sklearn.utils import shuffle
from self_Attention import Self_Attention
from keras.layers import concatenate,dot
import numpy as np
from data_helper_block5_pool import  load_data
from result_calculator import *
from keras import backend as K
from keras.utils.vis_utils import plot_model
from selfc import Position_Embedding,Attention
from selfDef import  coAttention_para, zero_padding, tagOffSet
from keras.utils import np_utils
import h5py

import os


if __name__ == '__main__':
    print ('loading data...')
    
    x, img_x, y, valid_x, valid_img_x, valid_y, test_x, test_img_x, test_y, vocabulary, vocabulary_inv, hashtagVoc, hashtagVoc_inv, maxlen = load_data()
    

   
    vocab_size = len(vocabulary_inv) + 1
    hashtag_size = len(hashtagVoc_inv)

    
    
    embedding_dim = 500
    nb_epoch = 100
    batch_size = 128

    feat_dim = 512
    w = 7

    # build model
    print ("Build model...")

    tweet = Input(shape=(maxlen,), dtype='int32')#input_1

    textEmbeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                               mask_zero=False, input_length=maxlen)
    lstm = LSTM(units=embedding_dim, return_sequences=True)
    dense = Dense(500, activation="tanh", use_bias=False)
    reshape = Reshape(target_shape=(49, 512))
    coAtt_layer = coAttention_para(dim_k=100)

    img = Input(shape=(7,7,512))#input_2 
    
 
    text_embeddings = textEmbeddings(tweet)
    tFeature = Self_Attention(500)(text_embeddings)
    tFeature=Dropout(0.5)(tFeature)
    
    iFeature = reshape(img)
    
    
    
    
    img_avg = AveragePooling1D(pool_size=49)(iFeature)
    img_avg = Flatten()(img_avg)
    img_avg = Dense(500)(img_avg)

    tweet_avg = AveragePooling1D(pool_size=maxlen)(tFeature)
    tweet_avg = Flatten()(tweet_avg)
    tweet_avg = Dense(500)(tweet_avg)

    im_tw_sum=Add()([img_avg,tweet_avg])
    print("-im_tw_sum-",im_tw_sum.shape)
    
    
    
    
    
    
    
    iFeature = dense(iFeature)
    co_feature = coAtt_layer([tFeature, iFeature])

  
    
    co_feature=Add()([co_feature,im_tw_sum])
   



    output = Dense(hashtag_size, activation='softmax')(co_feature)
    
    model = Model(inputs=[tweet, img], output=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        
    print (model.summary())
    print ("finished building model")
 
    y = np_utils.to_categorical(y, hashtag_size)
    
    
    print ("starts training")
    best_f1 = 0
    topK = [1, 2, 3, 4, 5]
    for j in range(nb_epoch):
    
        history = History()
    
    
       
    
            
        model.fit([x,img_x],y,batch_size=batch_size,epochs=1,verbose=1,callbacks=[history])
            
        print (history.history)
        print (len(history.history))
        y_pred = model.predict([valid_x, valid_img_x], batch_size=batch_size, verbose=1)
            
        y_pred = np.argsort(y_pred, axis=1)
            #argsort      
        precision = precision_score(valid_y, y_pred)
        recall = recall_score(valid_y, y_pred)
        F1 = 2 * (precision * recall) / (precision + recall)
        print ("Epoch:", (j + 1), "Valid_precision:", precision, "Valid_recall:", recall, "Valid_f1 score:", F1)
    
        if best_f1 < F1:
            best_f1 = F1
            y_pred = model.predict([test_x, test_img_x], batch_size=batch_size, verbose=1)
    
            y_pred = np.argsort(y_pred, axis=1)
            for k in topK:
                precision = precision_score(test_y, y_pred, k=k)
                recall = recall_score(test_y, y_pred, k=k)
                hscore = hits_score(test_y, y_pred, k=k)
                F1 = 2 * (precision * recall) / (precision + recall)
                print ("\t", "top:", k, "Test: precision:", precision, "recall:", recall, "f1 score:", F1, "hits score:", hscore)
    