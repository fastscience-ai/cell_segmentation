import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM

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
        for t in range(int(len(image)/timestep)):
            img = np.reshape(image[t*timestep:t*timestep+timestep], [1,timestep,70,70,1])
            lab = np.reshape(label[t*timestep:t*timestep+timestep],[1,timestep,70,70,3])
            #img = np.reshape(image[t:t+timestep], [1,timestep,int(size/4),int(size/4),1])
            #lab = np.reshape(label[t:t+timestep],[1,timestep,int(size/4),int(size/4),3])
            image_concat = []
            for k in range(2): #range(5):
                image_concat.append(np.multiply(img[:,0:1,:,:,:],lab[:,0:1,:,:,k+1:k+2]))
            img_0 = np.concatenate(image_concat, 4)
            img = np.concatenate([img for k in range(2)], 4)
            #img = np.concatenate([img for k in range(5)], 4)
            img = np.concatenate( [img_0, img_0,  img_0, img_0, img_0, img], 1)
            lab_0 = lab[:,0:1,:,:,:]
            lab = np.concatenate( [lab_0, lab_0,  lab_0, lab_0, lab_0, lab], 1)
            input_file.append(img)
            output_file.append(lab)
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
    return train_x, train_y




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

def main():
    convlstm = ConvLSTM_Seg()
    device=get_device()
    print(device)
    X=torch.zeros([10,15,70,70,2])
    X = torch.permute(X,(0,4,1,2,3))
    trouble_shooting_out , h, c = convlstm(X, None, None, None, None, None, None, None, None)
    print(np.shape(trouble_shooting_out)) #torch.Size([10, 3, 15, 70, 70])
    optim = torch.optim.Adam(convlstm.parameters(), lr= 0.0007)


    # Binary Cross Entropy, target pixel values either 0 or 1
    criterion = nn.MSELoss()#reduction="sum") #BCELoss()
    #nn.MSELoss 
    EPOCHS = 100
    hist_all = []
    
    for epoch in range(EPOCHS):
        train_loss = 0
        convlstm.train()
        cnt = 0
        for jj in range(9):
            print(jj)
            
            train_x, train_y = crop_video(False, 35, 70, 10,[ "Bcell_1"], jj)
            print(np.shape(train_x), np.shape(train_y))
            mm = len(train_y)
            train_x = torch.permute(torch.from_numpy(train_x),(0,4,1,2,3))
            train_y = torch.permute(torch.from_numpy(train_y),(0,4,1,2,3))
            for k in range(len(train_x)):
                output,  h, c = convlstm(train_x[k:k+1].float(), None, None, None, None, None, None, None, None)
                loss = criterion(output.flatten().float(), train_y[k:k+1].flatten().float())   
                loss.backward()
                optim.step()
                optim.zero_grad()
                #print("Epoches",epoch,"-", jj,"step",k, loss.item())
                cnt+=1
                train_loss += loss.item()
        train_loss /= float(cnt)
        print("TRAIN LOSS Epoch "+str(epoch)+ " : "+str(train_loss)) 
         
        val_loss = 0
        convlstm.eval()
        with torch.no_grad():
            test_x, test_y = crop_video(False, 35, 70, 10, ["Bcell_1"], 9)
            m=len(test_y)
            test_x = torch.permute(torch.from_numpy(test_x),(0,4,1,2,3))
            test_y = torch.permute(torch.from_numpy(test_y),(0,4,1,2,3))
            output, h, c = convlstm(test_x.float(), None, None, None, None, None, None, None, None)
            loss = criterion(output.flatten().float(), test_y.flatten().float())
            val_loss += loss.item()                                
        val_loss /= m 
        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
        if epoch %10 == 0:
            torch.save(convlstm.state_dict(), "./torch_cpu_convlstm_mse_"+str(epoch)+"_"+str(val_loss))
        hist_all.append([epoch, train_loss, val_loss])
        hist_save=np.asarray(hist_all)
        np.save("hist.npy", hist_save)



    
if __name__ == "__main__":
    main()
