from tensorflow.keras import Model
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
import pandas as pd
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.legend()
  plt.savefig("learning_curve.png")


def masking(folder='./data/Bcell_1/'):
    folder = Path(folder)
    cells = [d for d in os.listdir(folder) if (folder / d).is_dir() ]
    x_list = []
    y_list = []
    for sample in cells:
        print(sample)
        lac = imageio.volread(folder/sample/ f"{sample}_lac.tiff")
        labels = imageio.volread(folder/sample/ f"{sample}_labels.tiff")
        d1,d2,d3=np.shape(labels)
        masks = np.zeros([d1,d2,d3])
        labels_new =np.zeros([d1,d2,d3,5])
        #(1) make mask and new labels
        #Old anotation: "mitochondria": 2, "endoplasmic reticulum": 3, "lipid": 4, "heterochromatin": 5, "euchromatin": 6
        #New anotation: "mitochondria": 1st-ch, "endoplasmic reticulum": 2nd, "lipid": 3rd, "heterochromatin": 4th, "euchromatin": 5th
        start = 0
        end = 0
        count = 0
        for i in range(d1):
            on = False
            for j1 in range(d2):
                for j2 in range(d3):
                    if labels[i,j1,j2] >0:
                        masks[i,j1,j2] =1
                        ch = int(labels[i,j1,j2])-2
                        labels_new[i,j1,j2,ch] =1
            if np.amax(masks[i]) > 0:
                end = i
                count+=1
                if count ==1:
                    start = i
                lac[i] = np.multiply(lac[i],masks[i])
        x,y = np.asarray(lac[start:end]),np.asarray(labels_new[start:end])
        y.astype('b')
        print(np.shape(x), np.shape(y))
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

# explicit function to normalize array
def normalize(arr, t_min, t_max, max_val, min_val):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max_val - min_val
    for i in arr:
        temp = (((i - min_val)*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.asarray(norm_arr)



def crop_boundary(x,y):
    d1,d2,d3=np.shape(x)
    d21=1
    d22=d2-1
    d31=1 
    d32=d3-1
    print("before crop", type(x), type(y), np.shape(x), np.shape(y))
    check=True
    while(check):
        if np.max(x[:,0:d21,:])*np.max(x[:,d22:d2,:])*np.max(x[:,:,0:d31])*np.max(x[:,:,d32:d3]) >0:
            check = False
        if np.max(x[:,:d21,:])==0:
            d21+=1
        if np.max(x[:,d22:,:])==0:
            d22-=1
        if np.max(x[:,:,:d31])==0:
            d31+=1
        if np.max(x[:,:,d32:])==0:
            d32-=1
    return x[:,d21:d22,d31:d32], y[:,d21:d22,d31:d32]


def collect_data(save_fig, files):
    x_all=[]
    y_all=[]
    for name in files:
        print(name)
        xx,yy=masking("./data/"+name)
        for l in range(len(xx)):
            x_all.append(xx[l])
            y_all.append(yy[l])
    print(len(x_all), len(y_all))
    min_val=min([np.amin(x_all[i]) for i in range(len(x_all))])
    max_val=max([np.amax(x_all[i]) for i in range(len(x_all))])
    for i in range(len(x_all)):
        x_all[i] = normalize(x_all[i], 0, 1, max_val, min_val)
        print("after norm", max_val, min_val, i,np.min(x_all[i]), np.max(x_all[i]))
        #after norm 0.02874858 0.0 1 0.0 1.0
        #crop boundary
        x_all[i], y_all[i] = crop_boundary(x_all[i], y_all[i])
        print("after cropping",i, np.shape(x_all[i]), np.shape(y_all[i]))
        if save_fig:
            for j in range(len(x_all[i])):
                fig = plt.figure(figsize=(10, 5))
                fig.add_subplot(1, 2, 1)
                plt.imshow(x_all[i][j])
                fig.add_subplot(1, 2, 2)
                plt.imshow(y_all[i][j])
                plt.savefig("Bcell_"+str(i)+"_"+str(j)+".png")
                plt.close()
        np.save("x_"+str(i)+".npy", x_all[i])
        np.save("y_"+str(i)+".npy", y_all[i])
    return x_all, y_all



def crop_video(collect_model, stride, size, timestep, file_listi,jj):
    if collect_model:
        x_all, y_all = collect_data(False, file_list)
    else:
        x_all =[]
        y_all = []
        if True: #jj:0...15
           x_inner =  np.load("./input_data/x_"+str(jj)+".npy")
           y_inner = np.load("./input_data/y_"+str(jj)+".npy")
           x_all.append(x_inner)
           y_all.append(y_inner)
    x_t = []
    y_t = []
    for i in range(len(x_all)):
        x=x_all[i]
        y=y_all[i]
        d1,d2,d3=np.shape(x)
        for i2 in range(int((d2-size)/stride)+1):
            for i3 in range(int((d3-size)/stride)+1):
                if i2*stride+size > d2 and i3*stride+size < d3:
                    x_ = x[:, size:, i3*stride: i3*stride+size]
                    y_ = y[:, size:, i3*stride: i3*stride+size,:]
                if i2*stride+size > d2 and i3*stride+size > d3:
                    x_ = x[:, size:, size:]
                    y_ = y[:, size:,  size:,:]
                if i2*stride+size < d2 and i3*stride+size > d3:
                    x_ = x[:, i2*stride: i2*stride+size, size:]
                    y_ = y[:, i2*stride: i2*stride+size, size,:]
                else:
                    x_ = x[:, i2*stride: i2*stride+size, i3*stride: i3*stride+size]
                    y_ = y[:, i2*stride: i2*stride+size, i3*stride: i3*stride+size,:]    
                end = d1
                start = 0
                for i1 in range(d1):
                    if np.max(x[i1]) > 0.0:
                        start = i1
                        break
                for i1 in range(d1):
                    if np.max(x[d1-1-i1]) >0.0:
                        end = d1-i1
                        break
                #print(np.shape(x_[start:end]), np.shape(y_[start:end]))
                x_t.append(x_[start:end])
                y_t.append(y_[start:end])
    
    input_file = []
    output_file = []
    for image, label in list(zip(x_t[:], y_t[:])):
        for t in range(len(image)-timestep):
            #print(np.shape(image[t:t+timestep]), np.shape(label[t:t+timestep]))
            input_file.append(np.reshape(image[t:t+timestep], [1,timestep,size,size,1]))
            output_file.append(np.reshape(label[t:t+timestep],[1,timestep,size,size,5]))
    input_file_ =[]
    output_file_ = []
    idx = [i for i in range(len(input_file))]
    random.shuffle(idx)
    #print(idx, len(input_file))
    for k in idx:
        input_file_.append(input_file[k])
        output_file_.append(output_file[k])
    train_x = np.concatenate(input_file_, 0)
    train_y = np.concatenate(output_file_, 0)
    #print(np.shape(train_x), np.shape(train_y))
    return train_x, train_y



class ConvLSTM(Model):
  def __init__(self):
    super(ConvLSTM, self).__init__()
    self.layer1 = layers.ConvLSTM2D(filters=8, kernel_size=(3,3), input_shape=(10,200,200,1), padding="same", return_sequences=True)
    self.layer2 = layers.ConvLSTM2D(16, (5,5),  padding="same", return_sequences=True),
    self.layer3 = layers.ConvLSTM2D(16, (9,9),  padding="same", return_sequences=True),
    self.layer4 = layers.ConvLSTM2D(5, (12,12),  padding="same", return_sequences=True),

  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x

def main():

    convlstm =  Sequential([
        layers.ConvLSTM2D(filters=8, kernel_size=(3,3), input_shape=(5,50,50,1), padding="same", return_sequences=True),
        layers.ConvLSTM2D(16, (5,5),  padding="same", return_sequences=True),
        layers.ConvLSTM2D(32, (9,9),  padding="same", return_sequences=True),
        layers.ConvLSTM2D(5, (10,10),  padding="same", return_sequences=True),
        ])
    #test
    trouble_shooting_out = convlstm.predict(tf.zeros([10,5,50,50,1]))
    print(np.shape(trouble_shooting_out))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss_fn = 'mse'#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    convlstm.compile(loss=loss_fn, optimizer=optimizer, metrics=['mse'])
    print(convlstm.summary())
    
    EPOCHS = 100
    for epoch in range(EPOCHS):
        for jj in range(16):
            train_x, train_y = crop_video(False, 200, 200, 5,[ "Bcell_1", "Bcell_2"], jj)
            print(np.shape(train_y))# (987, 10, 200, 200) 
            history = convlstm.fit(
                    train_x[:,0:5,75:125,75:125,:], train_y[:,0:5,75:125,75:125],
                  epochs=2, validation_split = 0.1, verbose=1)
            convlstm.save("convlstm_in")
        convlstm.save("convlstm_"+str(epoch))
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())
        plot_history(history)
    #test
    test_x, test_y = crop_video(True, 200, 200, 10, ["Bcell_test"])
    #Evaluate with test set
    loss,  mse = convlstm.evaluate(test_x, test_y, verbose=2)
    print("MSE of testset: {:5.2f}".format(mse))

    test_result =[]
    for i in range(len(test_x)):
        test_predictions = convlstm.predict(test_x[i:i+1])
        test_result.append(test_predictions)
    test_results = np.concatenate(test_result, 0)
    np.save("model_prediction_y.npy", test_results)
    np.save("ground_truth_y.npy", test_y)
    np.save("ground_truth_x.npy", test_x)


    
if __name__ == "__main__":
    main()
