from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Reshape
import numpy as np
import tensorflow as tf
import h5py
import time
import cv2


weights = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
base_model = VGG16(weights=weights,include_top=True)
for i ,layer in enumerate(base_model.layers):
    print(i,layer.name,layer.output_shape)


dataPath = ""
filename = ''
outputname = ''
error = ''
f_w = open(error, 'a')
f_out = h5py.File(outputname,'a')
mean_pixel = [103.939, 116.779, 123.68]  

#Build model
model = Model(inputs= base_model.input,outputs=base_model.get_layer('block5_pool').output)
img_list = []
with open(filename) as f:
    for line in f:
        line = line.strip().split('\n')
        img_list.append(line[0])
    for item in img_list:
        #print(img_list)
        print ("process " + item + '.jpg')
        name = dataPath + "/"+item + '.jpg'
        #print(name)
        try:
            im = cv2.resize(cv2.imread(name),(224,224)).astype(np.float32)
           
        except:
            f_w.write(item + '\n')
            continue
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im/127.5-1
        x = image.img_to_array(im)
        x = x.transpose((1, 0, 2))
        x = np.expand_dims(x ,axis=0)
        #x = np.array(x,dtype=np.float32 )
        x = preprocess_input(x)


        start = time.time()
        features = model.predict(x)
        print(features[0].shape)
       
    

        print ('%s feature extracted in %f  seconds.'%('img_name',time.time()-start))

        f_out.create_dataset(name=item,data=features[0])



