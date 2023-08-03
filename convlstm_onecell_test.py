import sys
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
import tifffile
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

def downscale( image, label, size):
    #image : [n, 200, 200]
    #label:  [n, 200, 200, 2]
    d1,d2,d3 = np.shape(image)
    image_ = []
    label_ = []
    print(np.shape(image), np.shape(label))
    for i in range(d1):
        x = cv2.resize(image[i], dsize=(size,size), interpolation=cv2.INTER_LINEAR)
        x = np.resize(x, [1, size,size])
        y = cv2.resize(label[i], dsize=(size,size), interpolation=cv2.INTER_LINEAR)
        y = np.resize(y, [1, size,size, 3])
        image_.append(x)
        label_.append(y)
    image_ = np.concatenate(image_, 0)
    label_ = np.concatenate(label_,0)
    return image_, label_


def crop_video_test(collect_model, stride, size, timestep, file_list, jj, reverse):
    # if reverse == True: half to start
    # else: half to end
    size = size*4
    stride = stride*4
    if collect_model:
        x_all, y_all = collect_data(False, file_list)
    else:
        if True:
           x_inner =  np.load("./input_data/x_"+str(jj)+".npy")
           y_inner = np.load("./input_data/y_"+str(jj)+".npy")
           #print(np.shape(x_inner), np.shape(y_inner)) # (380, 644, 373) (380, 644, 373, 6)

           #############NOTE: SOOOO################
           cutting_idx = 190
           if reverse:
                x_inner = np.flip(x_inner[:cutting_idx],0)
                y_inner = np.flip(y_inner[:cutting_idx],0)
           else:
                x_inner = x_inner[cutting_idx:]
                y_inner = y_inner[cutting_idx:]
           x_all=[x_inner[:,:322,:], x_inner[:,322:,:]]
           y_all=[y_inner[:,:322,:,:], y_inner[:,322:,:,:]]
    train_x=[]
    train_y=[]
    idx_ = []
    for i in range(len(x_all)):
        start = False
        idx = 0
        while not start:
            if np.amax(y_all[i][idx,:,:,1:]) >0:
                start = True
            else:
                idx += 1
        #print(i, idx)
        idx_.append(idx)
        #print("before",i, np.shape(y_all[i]), np.shape(x_all[i]))
        x_all[i] = x_all[i][idx:]
        y_all[i] = y_all[i][idx:]
        #print("after_cutting", i, np.shape(x_all[i]), np.shape(y_all[i])) #(190, 322, 373) (190, 322, 373, 3)
    for image, label in list(zip(x_all, y_all)):
        input_file = []
        output_file = []
        image, label = downscale(image, label, 70)
        #NOTE: copy label to the input 
        for t in range(int(len(image)/timestep)):
            img = np.reshape(image[t*timestep:t*timestep+timestep], [1,timestep,70,70,1])
            lab = np.reshape(label[t*timestep:t*timestep+timestep],[1,timestep,70,70,3])
            image_concat = []
            for k in range(2):
                image_concat.append(np.multiply(img[:,0:1,:,:,:],lab[:,0:1,:,:,k+1:k+2]))
            img_0 = np.concatenate(image_concat, 4)
            img = np.concatenate([img for k in range(2)], 4)
            img = np.concatenate( [img_0, img_0,  img_0, img_0, img_0, img], 1)
            lab_0 = lab[:,0:1,:,:,:]
            lab = np.concatenate( [lab_0, lab_0,  lab_0, lab_0, lab_0, lab], 1)
            input_file.append(img)
            output_file.append(lab)
        train_x.append(np.concatenate(input_file,0))
        train_y.append(np.concatenate(output_file,0))
    return train_x, train_y, idx_



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
    convlstm = tf.keras.models.load_model('convlstm2_latest_ch2')
    # Check its architecture
    #print(convlstm.summary())
    pre_all = []
    gt_all = []
    input_img_all = []
    interval = int(sys.argv[1])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n")
    print("Feed ground truth segmentation mask as a supervision per every "+str(interval*10)+" timestep\n\n\n")
    for reverse in [True, False]:
        test_x, test_y, idx = crop_video_test(False, 70, 70, 10, ["Bcell_test"], 9, reverse)
        #print(np.shape(test_x), np.shape(test_y)) #(2, 19, 15, 70, 70, 2) (2, 19, 15, 70, 70, 3)
        #Evaluate with test set
        pre_ = []
        gt_ = []
        input_img_ = []
        #print(idx, np.shape(test_x[0]), np.shape(test_x[1]))
        for l in range(len(test_x)):
            pre = [np.zeros([idx[l],70,70,3])]
            gt = [np.zeros([idx[l],70,70,3])]
            input_img = [np.zeros([idx[l],70,70,2])]
            for i in range( len(test_x[l])):
                #if i >0: #if its from second batch
                #if False:
                if i !=0 and not((i+1)%interval) ==0:
                    #print(np.shape(test_predictions), np.shape(test_x[l][i:i+1]))#(1, 15, 70, 70) (1, 15, 70, 70, 2)
                    test_predictions = np.argmax(test_predictions, -1 )
                    prediction_1 = test_predictions[0,-1,:,:]==1
                    prediction_2 = test_predictions[0,-1,:,:]==2
                    first_shot_1 = np.multiply(test_x[l][i:i+1][0,5,:,:,0], prediction_1)
                    first_shot_2 = np.multiply(test_x[l][i:i+1][0,5,:,:,1], prediction_2)
                    for k in range(5):
                        #print(np.shape(test_x[l]), np.shape(first_shot_1))
                        test_x[l][i,k,:,:,0] = first_shot_1
                        test_x[l][i,k,:,:,1] = first_shot_2
                test_predictions = convlstm.predict(test_x[l][i:i+1])
            #    print(np.shape(test_x[l][i:i+1]), np.shape(test_predictions))
                pre.append(test_predictions[0,5:])
                gt.append(test_y[l][i:i+1][0,5:])
                input_img.append(test_x[l][i:i+1][0,5:])
            pre_.append(np.concatenate(pre,0))
            gt_.append(np.concatenate(gt, 0))
            input_img_.append(np.concatenate(input_img, 0))
           # print(np.shape(pre_[-1]), np.shape(gt_[-1]), np.shape(input_img_[-1]))
        m = min([len(pre_[i]) for i in range(len(pre_))])
        #print(m)
        pre_ = np.concatenate([pre_[i][:m] for i in range(len(pre_)) ],1)
        gt_ = np.concatenate([gt_[i][:m] for i in range(len(gt_))],1)
        input_img_ = np.concatenate([input_img_[i][:m] for i in range(len(input_img_))], 1) #combine up and down
        #print(reverse, np.shape(gt_), np.shape(pre_), np.shape(input_img_))  #(190, 140, 70, 3) (190, 140, 70, 3) (190, 140, 70, 2)
        if reverse:
            gt_ = np.flip(gt_,0) 
            pre_ = np.flip(pre_, 0)
            input_img_ = np.flip(input_img_, 0)
        pre_all.append(gt_) ; gt_all.append(pre_); input_img_all.append(input_img_)
    pre_ = np.concatenate(pre_all, 0)
    gt_ = np.concatenate(gt_all, 0)
    input_img_ = np.concatenate(input_img_all, 0)
    
    print(np.shape(pre_), np.shape(gt_), np.shape(input_img_))
    np.save("./fig_stamp/"+str(interval*10)+"/prediction_"+str(interval*10)+".npy", pre_)
    np.save("./fig_stamp/"+str(interval*10)+"/ground_truth_"+str(interval*10)+".npy", gt_)
    np.save("./fig_stamp/"+str(interval*10)+"/input_img_"+str(interval*10)+".npy", input_img_)
    tifffile.imwrite('./fig_stamp/'+str(interval*10)+'/prediction_'+str(interval*10)+'.tif', pre_)
    tifffile.imwrite('./fig_stamp/'+str(interval*10)+'/ground_truth_'+str(interval*10)+'.tif', gt_)
    tifffile.imwrite('./fig_stamp/'+str(interval*10)+'/input_img_'+str(interval*10)+'.tif', input_img_)
    for i in range(int(len(pre_)/10)):
          iou, dice_score = compute_metrics(np.argmax(pre_[i*10:(i+1)*10], -1), np.argmax(gt_[i*10:(i+1)*10], -1), 2)
          print(i, "IOU",  iou, "DICE SCORE", dice_score)
          fig = plt.figure(figsize=(20, 6))
          output_ch_size = 2
          rows = output_ch_size +2
          columns = 10
          for j1 in range(rows):
            if j1 < output_ch_size:
                img = input_img_[i*10:(i+1)*10][:,:,:,1:]
                #print(np.shape(img)) 
            elif j1 == output_ch_size:
                img = np.argmax(gt_[i*10:(i+1)*10],-1)
            elif j1 == output_ch_size+1:
                img = np.argmax(pre_[i*10:(i+1)*10],-1)
            for j2 in range(10):
                #print("row", j1+1, "colums", j2+1)
                fig.add_subplot(rows, columns, j1*10+j2+1)
                plt.imshow(img[j2])
                plt.axis('off')
                if j1 ==0  and j2 == 5:
                     plt.title(str(l)+"-"+str(i)+" IOU: ch1 "+str(round(iou[0]*100, 2))+" ,ch2 "+str(round(iou[1]*100,2))+"\n DICE:  ch1 "+str(round(dice_score[0]*100,2))+", ch2 "+str(round(dice_score[1]*100,2))+"   \n Inputs")
                if j1== output_ch_size and j2 == 4:
                    plt.title(" output (GT)")
                if j1 == output_ch_size+1 and j2 == 4:
                    plt.title(" output (prediction)")
          plt.savefig("./fig_stamp/"+str(interval*10)+"/results_"+str(l)+"_"+str(i)+".png")
   

if __name__ == "__main__":
    main()
