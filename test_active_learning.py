import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM
#Free GPU memory
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        #New anotation: "background":0, "mitochondria": 1, "endoplasmic reticulum": 2, "lipid": 3, "heterochromatin": 4, "euchromatin": 5
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
                  # ch0: background, ch1: heterochromatin, ch2: euchromatin
                  #  if labels[i,j1,j2] >0:
                  #      masks[i,j1,j2] =1
                  #      ch = int(labels[i,j1,j2])
                  #      if ch == 0 or ch ==1:
                  #         labels_new[i,j1,j2,0] =1
                  #      else:
                  #         labels_new[i,j1,j2,int(ch-1)] =1
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
    #label:  [n, 200, 200, 3]
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
    return image, label
#input output (70, 70) (70, 70, 3)--->(1,6,70,70,2), (6,70,70,3)
def make_data(x, y):
    n,n=np.shape(x)
    _,_,c = np.shape(y)
    img = np.reshape(x, [1,1,n,n,1])
    lab = np.reshape(y, [1,1,n,n,c])
    image_concat = []
    for k in range(2):
        image_concat.append(np.multiply(img[:,0:1,:,:,:],lab[:,0:1,:,:,k+1:k+2]))
    img_0 = np.concatenate(image_concat, 4)
    img = np.concatenate([img for k in range(2)], 4)
    img = np.concatenate( [img_0, img_0,  img_0, img_0, img_0, img], 1)
    lab_0 = lab[:,0:1,:,:,:]
    lab = np.concatenate( [lab_0, lab_0,  lab_0, lab_0, lab_0, lab], 1)
    return img, lab


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



class ConvLSTM_Seg(nn.Module):
    def __init__(self):
        super(ConvLSTM_Seg, self).__init__()
        self.layer1 = ConvLSTM(in_channels=2, out_channels=16,kernel_size=(3,3), padding='same', activation="relu", frame_size=(70,70))
        self.layer2 = ConvLSTM(in_channels=16, out_channels=24,kernel_size=(5,5), padding='same',activation="relu", frame_size=(70,70))
        self.layer3 = ConvLSTM(in_channels=24, out_channels=32, kernel_size=(9,9), padding='same',activation="relu", frame_size=(70,70))
        self.layer4 = ConvLSTM(in_channels=32, out_channels=3,kernel_size=(12,12), padding='same',activation="relu", frame_size=(70,70))
    def forward(self, X, H1, H2, H3, H4, C1, C2, C3, C4): 
        # Forward propagation through all the layers
        output, h1, c1 = self.layer1(X, H1, C1)
        output, h2, c2 = self.layer2(output, H2, C2)
        output, h3, c3 = self.layer3(output, H3, C3)
        output, h4, c4 = self.layer4(output, H4, C4)
        hidden_state = [h1,h2,h3,h4]
        cell_state = [c1,c2,c3,c4]
        return nn.Softmax()(output), hidden_state, cell_state



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



#https://www.kaggle.com/getting-started/140636
def free_gpu_cache():
    print("Initial GPU Usage") 
    torch.cuda.empty_cache()
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

def visualization(image_, label_, test_predictions_, i, iou_all, index): #(1, T, 70, 70, 2) (1, T, 70, 70, 3) (1, T, 70, 70, 3)
    output_ch_size = 2
    _,T,_,_,output_ch_size = np.shape(image_)
    fig = plt.figure(figsize=(T+5,6))
    rows = output_ch_size + 2 # 4
    columns = T
    for j1 in range(rows):
        if j1 < output_ch_size:
            img = image_[0,:,:,:, j1:j1+1]
             
        elif j1 == output_ch_size:
            img = np.argmax(label_[0,:,:,:,:],3)
        elif j1 == output_ch_size+1:
            img = np.argmax(test_predictions_[0],3)
        for j2 in range(T):
            
            fig.add_subplot(rows, columns, j1*T+j2+1) 
            if len(np.shape(img))==4:
                plt.imshow(img[j2][:,:,0])
            else:
                plt.imshow(img[j2])
            plt.axis('off')
            if j1 ==0 and j2==3:
                plt.title("Heterochromatin IOU: ")
            if j1==0 and j2 > 4:
                plt.title("T="+str(j2-4)+"("+str(round(iou_all[j2-5][1],2))+")")
            if j1==1 and j2==3:
                plt.title("Euchromatin IOU: ")
            if j1==1 and j2>4:
                plt.title("T="+str(j2-4)+"("+str(round(iou_all[j2-5][2],2))+")")
                          
            if j1== output_ch_size and j2==3:
                plt.title(" output (GT)")
            if j1 == output_ch_size+1 and j2==3:
                plt.title(" output (prediction)")
    plt.savefig("./fig_active_results/results_"+str(index)+"_"+str(i)+".png")



def main():
    convlstm = ConvLSTM_Seg()
    convlstm.load_state_dict(torch.load("./torch_cpu_convlstm_mse_50_0.0005932901735587787"))
    convlstm.eval()
  
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        free_gpu_cache()
        convlstm = convlstm.cuda()
        print("GPU usage after putting our model: ")
        gpu_usage()

    X=torch.zeros([10,15,70,70,2])
    X = torch.permute(X,(0,4,1,2,3))
    trouble_shooting_out , h, c = convlstm(X.float(), 
                                           None, None, None, None, None, None, None, None)
    print(np.shape(trouble_shooting_out)) #torch.Size([10, 3, 15, 70, 70])
    
    test_xx, test_yy = crop_video(False, 35, 70, 10, ["Bcell_test"], 9)
    print("input output", np.shape(test_xx), np.shape(test_yy)) #input output (317, 70, 70) (317, 70, 70, 3)
    #Evaluate with test set
    pre=[]
    gt = []
    input_img = []
    iou_list = []
    index = 1
    iou_threshold = 0.7
    with torch.no_grad():
        test_predictions = []
        image = []
        label = []
        iou_all =[]
        if os.path.exists("./fig_active_results"):
            shutil.rmtree("./fig_active_results")
        os.mkdir("./fig_active_results")
        for i in range(len(test_xx)):
            if i==0:
                test_x, test_y = make_data(test_xx[i], test_yy[i])
                print(np.shape(test_x), np.shape(test_y)) #(1, 6, 70, 70, 2) (1, 6, 70, 70, 3)
                test_x = torch.permute(torch.from_numpy(test_x),(0,4,1,2,3))
                test_y = torch.permute(torch.from_numpy(test_y),(0,4,1,2,3))
                #print(np.shape(test_x), np.shape(test_y)) #torch.Size([1, 2, 6, 70, 70]) torch.Size([1, 3, 6, 70, 70])
                out, h, c = convlstm(test_x[i:i+1].float(), None, None, None, None, None, None, None, None)
            elif sum([iou[i]<iou_threshold for i in range(len(iou))]) > 0.5:
                #Visualize current shot
                image_ = np.concatenate(image,1)
                label_ = np.concatenate(label,1)
                test_predictions_ = np.concatenate(test_predictions,1)
                visualization(image_, label_, test_predictions_, i, iou_all, index)
                print("\n\n\n START NEW:"+str(index)+"-th +++++++++++\n\n\n")
                index = index+1
                input_img.append(image_)
                gt.append(label_)
                pre.append(test_predictions_)
                iou_list.append(iou_all)
                test_predictions =[]; label=[]; image=[]; iou_all=[]
                test_x, test_y = make_data(test_xx[i], test_yy[i])
                test_x = torch.permute(torch.from_numpy(test_x),(0,4,1,2,3))
                test_y = torch.permute(torch.from_numpy(test_y),(0,4,1,2,3))
                out, h, c = convlstm(test_x[0:1].float(), None, None, None, None, None, None, None, None)
            else:
                test_x, test_y = make_data(test_xx[i], test_yy[i])
                test_x, test_y = test_x[:,5:, :,:,:], test_y[:,5:,:,:,:]
                assert(len(h) == len(c))
                h_ = []
                c_ = []
                for j in range(len(h)):
                    h_.append(np.asarray(h[j]))
                    c_.append(np.asarray(c[j]))
                test_x = torch.permute(torch.from_numpy(test_x),(0,4,1,2,3))
                test_y = torch.permute(torch.from_numpy(test_y),(0,4,1,2,3))
                #print(np.shape(test_x), np.shape(test_y))
                out, h, c = convlstm(test_x[0:1].float(),h[0].float(),h[1].float(),h[2].float(),h[3].float(), c[0].float(),c[1].float(),c[2].float(),c[3].float()) 
            out = torch.permute(out, (0,2,3,4,1))
            out = out.cpu().numpy()
            gt_y = torch.permute(test_y, (0,2,3,4,1))
            gt_y = gt_y.cpu().numpy()
            gt_x = torch.permute(test_x, (0,2,3,4,1))
            gt_x = gt_x.cpu().numpy()
            if i==0 or sum([iou[i]<iou_threshold for i in range(len(iou))]) > 0.5:
                _,k,_,_,_=np.shape(out)
                for j in range(k):
                    test_predictions.append(out[:,j:j+1,:,:,:])
                    image.append(gt_x[:,j:j+1,:,:,:])
                    label.append(gt_y[:,j:j+1,:,:,:])
            else:
                test_predictions.append(out)
                image.append(gt_x)
                label.append(gt_y)
            #print(np.shape(out), np.shape(gt_y), np.shape(gt_x)) #(1, 6, 70, 70, 3) (1, 6, 70, 70, 3)
            iou, dice_score = compute_metrics(np.argmax(out[0,-1,:,:,:],2), np.argmax(gt_y[0,-1,:,:,:],2),3)
            print("IOU:", iou)
            iou_all.append(iou)
        assert(len(pre) == len(gt) == len(input_img) == len(iou_list))


        




    
if __name__ == "__main__":
    main()
