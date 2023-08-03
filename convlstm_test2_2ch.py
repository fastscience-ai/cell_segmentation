import cv2
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
  plt.plot(hist['epoch'], hist['accuracy'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['accuracy'],
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
        labels_new =np.zeros([d1,d2,d3,3])
        #(1) make mask and new labels
        #Old anotation: "mitochondria": 2, "endoplasmic reticulum": 3, "lipid": 4, "heterochromatin": 5, "euchromatin": 6
        #New anotation: "background"(not "heterochromatin" or not "euchromatin"): 1th, "heterochromatin"or "euchromatin": 2nd
        start = 0
        end = 0
        count = 0
        for i in range(d1):
            on = False
            for j1 in range(d2):
                for j2 in range(d3):
                    if labels[i,j1,j2] >0:
                        masks[i,j1,j2] =1
                        ch = int(labels[i,j1,j2])
                        if ch ==5 :
                           labels_new[i,j1,j2,1] =1
                        elif ch == 6:
                           labels_new[i,j1,j2,2] =1
                        else:
                           labels_new[i,j1,j2,0] =1
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
        #print(name)
        xx,yy=masking("./data/"+name)
        for l in range(len(xx)):
            x_all.append(xx[l])
            y_all.append(yy[l])
    #print(len(x_all), len(y_all))
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
            print("saving figures")
            for j in range(len(x_all[i])):
                fig = plt.figure(figsize=(10, 5))
                fig.add_subplot(1, 2, 1)
                plt.imshow(x_all[i][j])
                fig.add_subplot(1, 2, 2)
                plt.imshow(y_all[i][j][:,:,1:2])
                plt.savefig("./fig/Bcell_"+str(i)+"_"+str(j)+".png")
                plt.close()
        np.save("./input_data/x_"+str(i)+".npy", x_all[i])
        np.save("./input_data/y_"+str(i)+".npy", y_all[i])
    return x_all, y_all

def downscale( image, label, mag):
    #image : [n, 200, 200]
    #label:  [n, 200, 200, 2]
    d1,d2,d3 = np.shape(image)
    image_ = []
    label_ = []
    for i in range(d1):
        x = cv2.resize(image[i], dsize=(int(d2/mag),int(d3/mag)), interpolation=cv2.INTER_LINEAR)
        x = np.resize(x, [1, int(d2/mag),int(d3/mag)])
        y = cv2.resize(label[i], dsize=(int(d2/mag),int(d3/mag)), interpolation=cv2.INTER_LINEAR)
        y = np.resize(y, [1, int(d2/mag),int(d3/mag), 3])
        image_.append(x)
        label_.append(y)
    image_ = np.concatenate(image_, 0)
    label_ = np.concatenate(label_,0)
    return image_, label_


def crop_video(collect_model, stride, size, timestep, file_list,jj):
    size = size*4
    stride = stride*4
    if collect_model:
        x_all, y_all = collect_data(False, file_list)
    else:
        x_all =[]
        y_all = []
        if True:
           x_inner =  np.load("./input_data/x_"+str(jj)+".npy")
           y_inner = np.load("./input_data/y_"+str(jj)+".npy")
           print(np.shape(x_inner), np.shape(y_inner))
           x_all.append(x_inner)
           y_all.append(y_inner)
    x_t = []
    y_t = []
    for i in range(len(x_all)):
        x=x_all[i]
        y=y_all[i]
        d1,d2,d3=np.shape(x)
        #size =  size*4 
        for i2 in range(int((d2-size)/stride)+1):
            for i3 in range(int((d3-size)/stride)+1):
                if i2*stride+size > d2 and i3*stride+size < d3:
                    try:
                        x_ = x[:, size:, i3*stride: i3*stride+size]
                        y_ = y[:, size:, i3*stride: i3*stride+size,:]
                    except:
                        pass
                if i2*stride+size > d2 and i3*stride+size > d3:
                    try:
                        x_ = x[:, size:, size:]
                        y_ = y[:, size:,  size:,:]
                    except:
                        pass
                if i2*stride+size < d2 and i3*stride+size > d3:
                    try:
                        x_ = x[:, i2*stride: i2*stride+size, size:]
                        y_ = y[:, i2*stride: i2*stride+size, size,:]
                    except:
                        pass
                else:
                    try:
                       x_ = x[:, i2*stride: i2*stride+size, i3*stride: i3*stride+size]
                       y_ = y[:, i2*stride: i2*stride+size, i3*stride: i3*stride+size,:]
                    except:
                        pass
                end = d1
                start = 0
                for i1 in range(d1):
                    if np.max(y[i1][:,:,1]) > 0.0:
                        start = i1
                        break
                for i1 in range(d1):
                    if np.max(y[d1-1-i1][:,:,1]) >0.0:
                        end = d1-i1
                        break
                #print(np.shape(x_[start:end]), np.shape(y_[start:end]))
                x_t.append(x_[start:end])
                y_t.append(y_[start:end])
    input_file = []
    output_file = []
    for image, label in list(zip(x_t[:], y_t[:])):
        image, label = downscale(image, label, 4)
        #NOTE: copy label to the input
        for t in range(len(image)-timestep):
            img = np.reshape(image[t:t+timestep], [1,timestep,int(size/4),int(size/4),1])
            lab = np.reshape(label[t:t+timestep],[1,timestep,int(size/4),int(size/4),3])
            image_concat=[]
            for k in range(2):
                image_concat.append(np.multiply(img[:,0:1,:,:,:],lab[:,0:1,:,:,k+1:k+2]))
            img_0 = np.concatenate(image_concat, 4)
            img = np.concatenate([img for k in range(2)], 4)
            img = np.concatenate( [img_0, img_0,  img_0, img_0, img_0, img], 1)
            lab_0 = lab[:,0:1,:,:,:]
            lab = np.concatenate( [lab_0, lab_0,  lab_0, lab_0, lab_0, lab], 1)

            input_file.append(img)
            output_file.append(lab)
    input_file_ =[]
    output_file_ = []
    idx = [i for i in range(len(input_file))]
    #random.shuffle(idx)
    #print(idx, len(input_file))
    for k in idx:
        input_file_.append(input_file[k])
        output_file_.append(output_file[k])
    train_x = np.concatenate(input_file_, 0)
    train_y = np.concatenate(output_file_, 0)
    return train_x, train_y



def compute_metrics(y_true, y_pred, number_of_label):
    '''
    Computes IOU and Dice Score.
 
    Args:
      y_true (tensor) - ground truth label map
      y_pred (tensor) - predicted label map
    '''
    class_wise_iou = []
    class_wise_dice_score = []
    smoothening_factor = 0.00001
    for i in range(number_of_label):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
 
        iou = (intersection + smoothening_factor) / \
            (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)
        dice_score = 2 * ((intersection + smoothening_factor) /
                          (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)
    return class_wise_iou, class_wise_dice_score


def main():
    convlstm = tf.keras.models.load_model('convlstm2_latest_ch2')#'convlstm2_latest_good')
    # Check its architecture

    print(convlstm.summary())
    #Generate data
    #for jj in range(9):
    #    train_x, train_y = crop_video(True, 35, 70, 10,[ "Bcell_1"], jj)
    #test
    test_x, test_y = crop_video(False, 35, 70, 10, ["Bcell_test"], 9)
    print(np.shape(test_x), np.shape(test_y))
    #Evaluate with test set
    for i in range( len(test_x)):
        test_predictions = convlstm.predict(test_x[i:i+1])
        test_predictions = test_predictions[:,:,:,:,:]
        iou, dice_score = compute_metrics(np.argmax(test_y[i:i+1][0, -1], -1), np.argmax(test_predictions[0, -1], -1), 2)
        print(i, "IOU",  iou, "DICE SCORE", dice_score, np.shape(test_x[i:i+1]), np.shape(test_y[i:i+1]), np.shape(test_predictions))
        #[1.423055554380401, 5.047955552509059e-09] [1.17459589685361, 1.0095911105018117e-08] (1, 10, 60, 60, 1) (1, 10, 60, 60, 2) (1, 10, 60, 60, 1)
        fig = plt.figure(figsize=(20, 6))
        output_ch_size = 2
        rows = output_ch_size +2
        columns = 15
        for j1 in range(rows):
            if j1 < output_ch_size:
                img = test_x[i,:,:,:, j1:j1+1]
                print(np.shape(img)) 
            elif j1 == output_ch_size:
                img = np.argmax(test_y[i,:,:,:,:],3)
            elif j1 == output_ch_size+1:
                print(np.shape(test_predictions))
                img = np.argmax(test_predictions[0],3)
            for j2 in range(15):
                print("row", j1+1, "colums", j2+1)
                fig.add_subplot(rows, columns, j1*15+j2+1)
                plt.imshow(img[j2])
                plt.axis('off')
                if j1 ==0  and j2 == 7:
                     plt.title("IOU: ch1 "+str(round(iou[0]*100, 2))+" ,ch2 "+str(round(iou[1]*100,2))+"\n DICE:  ch1 "+str(round(dice_score[0]*100,2))+", ch2 "+str(round(dice_score[1]*100,2))+"   \n Inputs")
                if j1== output_ch_size and j2 == 7:
                    plt.title(" output (GT)")
                if j1 == output_ch_size+1 and j2 == 7:
                    plt.title(" output (prediction)")
        plt.savefig("./fig_results_10/results_"+str(i)+".png")




if __name__ == "__main__":
    main()
