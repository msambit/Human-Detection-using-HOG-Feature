import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
import os
import imageio
import sys
import glob

# Loading Images from gdrive
from google.colab import drive
drive.mount('/content/drive')

# Writing output images into gdrive
directory = r'/content/drive/MyDrive/HumanDetection/'
os.chdir(directory)
np.set_printoptions(suppress=True)

def gradient(img):

    # Setting NxM matrices of Gradient Magnitude, Horizontal and Vertical Gradient to 0
    grad_mag = np.zeros(img.shape)
    Gx = np.zeros(img.shape)
    Gy = np.zeros(img.shape)

    size = img.shape

    # Prewitt's Operator
    kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

    # Calculating Horizontal and Vertical Gradients for all pixel values
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            Gx[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernelx))
            Gy[i, j] = np.sum(np.multiply(img[i - 1 : i + 2, j - 1 : j + 2], kernely))

    # Calculating gradient magnitude
    grad_mag = np.sqrt(np.add(np.square(Gx), np.square(Gy)))

    # Normalize gradient magnitude
    grad_mag = np.round((grad_mag/grad_mag.max())*255)

    # Calculating gradient angles
    angles = np.rad2deg(np.arctan2(Gy, Gx))
    angles[angles < 0 ] += 360

    return Gx, Gy, grad_mag, angles

def cell_hog(img, grad, angles):

  bin = np.zeros(10)

  # Segregating into Histogram Bins
  for i in range(0, 8):
        for j in range(0, 8):
          if(0 <= angles[i,j] < 10):
            bin[9] += (grad[i,j] * ((10 - angles[i,j])/20))
            bin[1] += (grad[i,j] * ((20 - (10 - angles[i,j]))/20))
          elif(10 <= angles[i,j] < 30):
            bin[1] += (grad[i,j] * ((30 - angles[i,j])/20))
            bin[2] += (grad[i,j] * ((angles[i,j] - 10)/20))
          elif(30 <= angles[i,j] < 50):
            bin[2] += (grad[i,j] * ((50 - angles[i,j])/20))
            bin[3] += (grad[i,j] * ((angles[i,j] - 30)/20))
          elif(50 <= angles[i,j] < 70):
            bin[3] += (grad[i,j] * ((70 - angles[i,j])/20))
            bin[4] += (grad[i,j] * ((angles[i,j] - 50)/20))
          elif(70 <= angles[i,j] < 90):
            bin[4] += (grad[i,j] * ((90 - angles[i,j])/20))
            bin[5] += (grad[i,j] * ((angles[i,j] - 70)/20))
          elif(90 <= angles[i,j] < 110):
            bin[5] += (grad[i,j] * ((110 - angles[i,j])/20))
            bin[6] += (grad[i,j] * ((angles[i,j] - 90)/20))
          elif(110 <= angles[i,j] < 130):
            bin[6] += (grad[i,j] * ((130 - angles[i,j])/20))
            bin[7] += (grad[i,j] * ((angles[i,j] - 110)/20))
          elif(130 <= angles[i,j] < 150):
            bin[7] += (grad[i,j] * ((150 - angles[i,j])/20))
            bin[8] += (grad[i,j] * ((angles[i,j] - 130)/20))
          elif(150 <= angles[i,j] < 170):
            bin[8] += (grad[i,j] * ((170 - angles[i,j])/20))
            bin[9] += (grad[i,j] * ((angles[i,j] - 150)/20))
          else:
            bin[9] += (grad[i,j] * ((20-(angles[i,j] - 170))/20))
            bin[1] += (grad[i,j] * ((angles[i,j] - 170)/20))
  
  return bin[1:]

def block_hog(img, grad, angles):

  hog = np.array([])

  for i in range(0, 2):
        for j in range(0, 2):
          hog = np.append(hog, cell_hog(img[i*8:i*8+8, j*8:j*8+8], grad[i*8:i*8+8, j*8:j*8+8], angles[i*8:i*8+8, j*8:j*8+8]))
           
  hog = hog.reshape(36,1)
  
  # Block Normalization
  k = np.sqrt(np.sum(np.square(hog)))
  normalizedVector = np.divide(hog, k, out=np.zeros_like(hog), where=k!=0)

  return normalizedVector

def hog_feature(img, grad, angles):

  vector = np.array([])

  for i in range(0, 19):
        for j in range(0, 11):
          vector = np.append(vector, block_hog(img[i : i + 16, j:j + 16], grad[i : i + 16, j:j + 16], angles[i : i + 16, j:j + 16]))

  vector = vector.reshape(7524, 1)

  return vector

def distance(featureI, featureM):
  total_dist=0.0
  total_featureM=0.0
  for i in range(featureI.shape[0]):
      total_dist += min(featureI[i][0],featureM[i][0])
      total_featureM += featureM[i][0]
      
  total=total_dist/total_featureM
  return total

def predict(X,data,test_name):

    pred=[]
    fname={}
    for i in range(len(X)):
        dist=[]
        dicti={}
        t_name=test_name[i]

        # Gradient Calculation of Input Image
        Gx, Gy, grad, angles=gradient(X[i])
        angles[angles > 180] -= 180

        cv2.imwrite(t_name, grad)

        # HOG Feature of Input Image
        feature_vectorI=hog_feature(X[i], grad, angles)

        with open('/content/drive/My Drive/HumanDetection/TestFeatures/'+t_name+'.txt', 'w') as Z:
          np.savetxt(Z, feature_vector, fmt='%.18f')

        # 3-nearest neighbor (NN) classifier
        for d,l,f in data:            
            dist.append((distance(feature_vectorI,d),l,f))

        # Three nearest neighbors
        dist=sorted(dist, key=lambda x: x[0], reverse=True)[:3]        
        print('Test Image - |', t_name, '| Neighbours - ', dist)
        fname[t_name]=dist

        # Majority classification
        for val in dist:
            if val[1] not in dicti:
                dicti[val[1]]=1
            else:
                dicti[val[1]]+=1
        label=sorted(dicti.items(), key=lambda x: x[1], reverse=True)
        pred.append(label[0][0])        
        
    pred=np.array(pred)
    return pred,fname

def train(X, y, f_name):
  data=[]
  for i,j,k in zip(X,y,f_name):
    data.append([i,j,k])
  return data

if __name__ == '__main__':

  x_train=[] # All Training Images
  y_train=[] # Training Image Classifications
  x_test=[] # All Test Images
  y_test=[] # Test Image Classifications
  f_name_train=[] #File names of training images
  f_name_test=[] #File names of test images

  dir=os.listdir('/content/drive/MyDrive/HumanDetection/Image Data')

  # Gathering all Training and Test Images with their labels
  for d in range(len(dir)):
    for i in glob.glob('/content/drive/MyDrive/HumanDetection/Image Data/'+dir[d]+'/*.bmp'):
      inputImg = imageio.imread(i, pilmode='RGB')
      # Converting Color Images to grayscale using formula
      inputImg = np.round(np.multiply(inputImg[:, :, 0], 0.299) + np.multiply(inputImg[:, :, 1], 0.587) + np.multiply(inputImg[:, :, 2], 0.114))
      if 'Training' in i and 'Neg' in i:
        x_train.append(inputImg)
        y_train.append(0)
        f_name_train.append(os.path.basename(i))
      elif 'Training' in i and 'Pos' in i:
        x_train.append(inputImg)
        y_train.append(1)
        f_name_train.append(os.path.basename(i))
      elif 'Test' in i and 'Neg' in i:
        x_test.append(inputImg)
        y_test.append(0)
        f_name_test.append(os.path.basename(i))
      else:
        x_test.append(inputImg)
        y_test.append(1)
        f_name_test.append(os.path.basename(i))

  x_train=np.asarray(x_train)
  x_test=np.asarray(x_test)
  y_train=np.asarray(y_train)
  y_test=np.asarray(y_test)
  
  data=train(x_train,y_train,f_name_train)

  feature_vectorData=[]

  # HOG Feature of all Training Images
  # Detection Window = 160x96; cell size = 8 x 8 pixels; block size = 16 x 16 pixels (or 2 x 2 cells); block overlap or step size = 8 pixels (or 1 cell.); 9 bins (unsigned)
  # non-overlapping cells = 20 x 12 cells; overlapping cells = 19 x 11 blocks
  for d,l,f in data:
    # Gradient Calculation
    Gx, Gy, grad, angles=gradient(d)
    angles[angles > 180] -= 180

    # HOG Feature
    feature_vector = hog_feature(d, grad, angles)

    with open('/content/drive/My Drive/HumanDetection/TrainingFeatures/'+f+'.txt', 'w') as Z:
      np.savetxt(Z, feature_vector, fmt='%.18f')

    #print(np.sum(feature_vector))
    feature_vectorData.append((feature_vector,l,f))

  y_pred,fname=predict(x_test,feature_vectorData,f_name_test)

print("Predicted:", y_pred)
print("Actual:   ", y_test)