import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from skimage import feature
from sklearn.ensemble import RandomForestRegressor
import scipy 

files=sorted(os.listdir('./MSRC_ObjCategImageDatabase_v1'))
def files_a(folder,files):
    pics=[]
    pixs=[]
    for i in range(0,len(files)):
        if files[i].endswith('GT.bmp'):
            pixs+=[files[i]]
        else:
            pics+=[files[i]]
    return pics,pixs
pics,pixs=files_a('MSRC_ObjCategImageDatabase_v1/',files)   


def rgb_features(folder,pics):
    """
    makes features of rgb
    """
    l=[]
    for i in pics:
        img=cv2.imread(folder+str(i))
        #print(img)
        img=img.astype(np.int32)
        #print('orig {}'.format(img.shape))
        img=scipy.misc.imresize(img,(150,150,3))
        l+=[img]
    return np.array(l)

def rgb_labelpixels(folder,pixs):
    """
    prepares labels 

    """
    t=[]
    for i in pixs:
        img=cv2.imread(folder+str(i))
        img=img.astype(np.int32)
        img=scipy.misc.imresize(img,(150,150,3))
        
        t+=[img]
    return np.array(t)


def mean(arr):
    return (arr[0]*1+arr[1]*2+arr[2]*3)/3


pics_f=rgb_features('MSRC_ObjCategImageDatabase_v1/',pics).reshape(-1,3)
print(pics_f.shape)
pixs_l=rgb_labelpixels('MSRC_ObjCategImageDatabase_v1/',pixs).reshape(-1,3)
print(pixs_l.shape)
pixs_l=np.apply_along_axis(mean,1,pixs_l)
#print('meaned pixels')
#for i in pixs_l:
#    print(i)

    



rgb_labels=[[0,128,0],[0,0,0],[0,0,128],[64,0,0],[0,128,128],[128,128,128],[128,0,0],[64,0,128],[128,0,128],[128,128,0],[192,0,128],[64,128,0]]

##output for referring

"""
[  42.66666667    0.           42.66666667   21.33333333   85.33333333
  128.           42.66666667   64.           85.33333333   85.33333333
  106.66666667   64.        ]
"""
classes=['grass','void','cow','mountain','sheep','sky','building','car','horse','tree','bicycle','water']

def mean(arr):
    return (arr[0]*1+arr[1]*2+arr[2]*3)/3


def rgb_mean(rgb_labels):
    arr=np.array(rgb_labels)
    arr=np.apply_along_axis(mean,1,arr)
    return arr

rgb_mean_labels=rgb_mean(rgb_labels)
print(rgb_mean_labels)
print(rgb_mean_labels.shape)


def map_to_label_classes(pixs_l):
    new_pixs_l=[]
    for k in pixs_l:
        m=np.abs(rgb_mean_labels-k)
        i=np.argmin(m)
        new_pixs_l+=[rgb_mean_labels[i]]
    return new_pixs_l
pixs_l=map_to_label_classes(pixs_l)


#for j in pixs_l:
#    print(pixs_l)
#print('shape'+str(pixs_l.shape))

clf=RandomForestRegressor(max_depth=11)
clf.fit(pics_f,pixs_l)

testfiles=os.listdir('./testdata')
test_pics,test_pixs=files_a('testdata/',testfiles)

#print(len(test_pics))
#print(len(test_pixs))
test_pics=rgb_features('testdata/',test_pics).reshape(-1,3)
#print(test_pics.shape)
test_pixs=rgb_labelpixels('testdata/',test_pixs).reshape(-1,3)
test_pixs=np.apply_along_axis(mean,1,test_pixs)
#print(test_pixs.shape)
y_pred=clf.predict(test_pics)
accur=clf.score(test_pics,test_pixs)
print(accur)

def generate_image_from_pixels(pixels):
    img=[]
    for j in range(100):
        for i in pixel:

             
            index=np.where(rgb_mean_labels==i)
            img.append(rgb_labels[index])
    return img

semantic_segmented_image=generate_image_from_pixels(test_pixs[0:22500]) # first image
cv2.imshow('IMAGE ',semantic_segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

