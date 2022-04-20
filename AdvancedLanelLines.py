#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Importing Libraries

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


# ### Step 1- Camera Calibration

# In[29]:


# Step 1- Camera calibration


#Creating an array for object Points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


objpoints=[] #3D points in real space 
imgpoints=[] #2D points in img space

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

f, axes= plt.subplots(1,2,figsize=(30,30))

for index,image in enumerate(images):
    originalImage= cv2.imread(image)
    grayImg= cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY) #converting to Grayscale before finding Chessboard Corners

    if(index==1 ):
        # Plotting the original Image
        axes[0].set_title('Original Image', fontsize=20)
        axes[0].imshow(originalImage)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(grayImg, (9,6), None)

    if(ret==True):
        objpoints.append(objp)
        imgpoints.append(corners)

        # Drawing Chessboard Corners
        cv2.drawChessboardCorners(originalImage, (9,6), corners, ret)
        if(index==1 ):
            axes[1].set_title('Image with Chessboard Corners', fontsize=20)
            axes[1].imshow(originalImage)

# from Step 1 we get the Object Points and Image Points


# ### Step 2- Calculate Undistortion Parameters 

# In[30]:


# Step 2- Calculating Undistortion Parameters

img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

f, axes= plt.subplots(1,2,figsize=(30,30))


axes[0].imshow(img)
axes[0].set_title("Original Image", fontsize=20)
axes[1].imshow(dst)
axes[1].set_title("Undistorted Images", fontsize=20)

#from Step 2 we get two important parameters- dist(the distortion coefficient), mtx(camera matrix)


# ### Step 3- Undistort Images

# In[31]:


# Step 3- Defining a function to undistort Images using parameters derived from previous step

def undistortImage(image):
    return cv2.undistort(image, mtx, dist, None, mtx)


# In[32]:


# Undistorting Test Images

f, axes= plt.subplots(8,2,figsize=(15,30))
f.subplots_adjust(hspace=0.5)

images = glob.glob('test_images/*.jpg') # Reading Images from test_images folder
original_untouched_images=[]

for index, image in enumerate(images):
    originalImage= cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    original_untouched_images.append(originalImage)
    axes[index,0].imshow(originalImage)
    axes[index,0].set_title("Original Image")
    undistortedImg=undistortImage(originalImage) # undistorting image 
    axes[index,1].set_title("Undistorted Image")
    axes[index,1].imshow(undistortedImg)


# ### Step 4, 5- ROI and Warping

# In[33]:


# Step 4, 5- Defining a Region of Interest, Warping an Image from bird's eye view

left=[150,720] #left bottom most point of trapezium
right=[1250,720] #right bottom most point of trapezium
apex_left=[590,450] # left top most point of trapezium
apex_right=[700,450] # right top most point of trapezium

src=np.float32([left,apex_left,apex_right,right]) # Source Points for Image Warp
dst= np.float32([[200 ,720], [200  ,0], [980 ,0], [980 ,720]]) # Destination Points for Image Warp


def ROI(originalImage):
    return cv2.polylines(originalImage,np.int32(np.array([[left,apex_left,apex_right,right]])),True,(0,0,255),10)

def WarpPerspective(image):
    y=image.shape[0]
    x=image.shape[1]
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (x,y), flags=cv2.INTER_LINEAR)


# In[34]:


# Testing ROI and Wrap on Test Images

f, axes= plt.subplots(8,3,figsize=(15,30))
f.subplots_adjust(hspace=0.5)

warpedImages=[]
for index, image in enumerate(images):
    originalImage= cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    untouchedImage= originalImage.copy()
    axes[index,0].imshow(originalImage)
    axes[index,0].set_title("Original Image")
    ROI(originalImage)
    axes[index,1].imshow(originalImage)
    axes[index,1].set_title("Image with Region Of Interest")
    y=untouchedImage.shape[0]
    x=untouchedImage.shape[1]
    warped = WarpPerspective(untouchedImage)
    warpedImages.append(warped)
    axes[index,2].imshow(warped)
    axes[index,2].set_title("Warped Image")
    
    # ### Step 6- Color Space

# In[35]:


# Step 6- Selecting a Color Space

def ExtractChannel(image,colorspace,threshold,channel=0):
    colorspace = cv2.cvtColor(image, colorspace)
    extracted_channel = colorspace[:,:,channel]
    binary = np.zeros_like(extracted_channel)
    binary[(extracted_channel >= threshold[0]) & (extracted_channel <= threshold[1])] = 1
    return binary


# In[36]:


# Testing Color Spaces on Test Images

f, axes= plt.subplots(5,4,figsize=(15,15))
#f.subplots_adjust(hspace=0.5)

image=warpedImages[0]
#for index,image in enumerate(warpedImages[0:1]):

threshold= [100,255]
index=0
axes[index,0].imshow(image)
axes[index,0].set_title("Original Image")

axes[index+1,0].imshow(image)
axes[index+1,0].set_title("Original Image")

axes[index+2,0].imshow(image)
axes[index+2,0].set_title("Original Image")

axes[index+3,0].imshow(image)
axes[index+3,0].set_title("Original Image")

axes[index+4,0].imshow(image)
axes[index+4,0].set_title("Original Image")


# HLS Colorspace
h=ExtractChannel(image, cv2.COLOR_RGB2HLS ,threshold,0)
axes[index,1].imshow(h,cmap='gray')
axes[index,1].set_title("H")

l=ExtractChannel(image, cv2.COLOR_RGB2HLS ,threshold,1)
axes[index,2].imshow(l,cmap='gray')
axes[index,2].set_title("L")

s=ExtractChannel(image, cv2.COLOR_RGB2HLS ,threshold,2)
axes[index,3].imshow(s,cmap='gray')
axes[index,3].set_title("S")

# HSV Colorspace
h=ExtractChannel(image, cv2.COLOR_RGB2HSV ,threshold,0)
axes[index+1,1].imshow(h,cmap='gray')
axes[index+1,1].set_title("H")

s=ExtractChannel(image, cv2.COLOR_RGB2HSV ,threshold,1)
axes[index+1,2].imshow(s,cmap='gray')
axes[index+1,2].set_title("S")

v=ExtractChannel(image, cv2.COLOR_RGB2HSV ,threshold,2)
axes[index+1,3].imshow(v,cmap='gray')
axes[index+1,3].set_title("V")

# YUV Colorspace
y=ExtractChannel(image, cv2.COLOR_RGB2YUV ,threshold,0)
axes[index+2,1].imshow(y,cmap='gray')
axes[index+2,1].set_title("Y")

u=ExtractChannel(image, cv2.COLOR_RGB2YUV ,threshold,1)
axes[index+2,2].imshow(u,cmap='gray')
axes[index+2,2].set_title("U")

v=ExtractChannel(image, cv2.COLOR_RGB2YUV ,threshold,2)
axes[index+2,3].imshow(v,cmap='gray')
axes[index+2,3].set_title("V")

# LAB Colorspace
l=ExtractChannel(image, cv2.COLOR_RGB2LAB ,threshold,0)
axes[index+3,1].imshow(l,cmap='gray')
axes[index+3,1].set_title("L")

a=ExtractChannel(image, cv2.COLOR_RGB2YUV ,threshold,1)
axes[index+3,2].imshow(a,cmap='gray')
axes[index+3,2].set_title("A")

b=ExtractChannel(image, cv2.COLOR_RGB2YUV ,threshold,2)
axes[index+3,3].imshow(b,cmap='gray')
axes[index+3,3].set_title("B")

# YCrCb Colorspace
y=ExtractChannel(image, cv2.COLOR_RGB2YCrCb ,threshold,0)
axes[index+4,1].imshow(y,cmap='gray')
axes[index+4,1].set_title("Y")

cr=ExtractChannel(image, cv2.COLOR_RGB2YCrCb ,threshold,1)
axes[index+4,2].imshow(cr,cmap='gray')
axes[index+4,2].set_title("Cr")

cb=ExtractChannel(image, cv2.COLOR_RGB2YCrCb ,threshold,2)
axes[index+4,3].imshow(cb,cmap='gray')
axes[index+4,3].set_title("Cb")


# ### Sobel

# In[37]:


# Step 7- Applying Sobel to warped image

def Sobel(warpedimage, threshold, sobelType, kernelSize=3):
    
    gray = cv2.cvtColor(warpedimage, cv2.COLOR_RGB2GRAY) # Step 1- Convert to GrayScale
    sobelx = cv2.Sobel(gray,cv2.CV_64F, 1, 0, ksize=kernelSize)
    sobely = cv2.Sobel(gray,cv2.CV_64F, 0, 1, ksize=kernelSize)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    grad= np.sqrt(sobelx**2 + sobely**2)
    
    arctan= np.arctan2(abs_sobely,abs_sobelx)
    
    valParam=abs_sobelx
    
    if(sobelType=='x'):
        valParam=abs_sobelx
    elif(sobelType=='y'):
        valParam= abs_sobely
    elif(sobelType=='xy'):
        valParam= grad
    else:
        valParam=arctan
        
    
    img = np.uint8((valParam* 255)/np.max(valParam)) # Creating a normalized sobel image
    binary_output = np.zeros_like(img)
    binary_output[(img > threshold[0]) & (img < threshold[1])]=1
    return binary_output


# In[38]:


# testing sobel on test_image and warped image
f, axes= plt.subplots(2,5,figsize=(20,8))

threshold=[20,100]
originalImage= images[0]
originalImage= cv2.cvtColor(cv2.imread(originalImage), cv2.COLOR_BGR2RGB)
index=0

axes[index,0].imshow(image)
axes[index,0].set_title("Warped Image")

sobelx=Sobel(image,threshold,'x')
axes[index,1].imshow(sobelx)
axes[index,1].set_title("Sobel X")

sobely=Sobel(image,threshold,'y')
axes[index,2].imshow(sobely)
axes[index,2].set_title("Sobel Y")

sobelxy=Sobel(image,threshold,'xy')
axes[index,3].imshow(sobelxy)
axes[index,3].set_title("Sobel Magnitude")

sobeldir=Sobel(image,threshold,'dir')
axes[index,4].imshow(sobeldir)
axes[index,4].set_title("Sobel Direction")

index=index+1

axes[index,0].imshow(originalImage)
axes[index,0].set_title("Original Image")

sobelx=Sobel(originalImage,threshold,'x')
axes[index,1].imshow(sobelx)
axes[index,1].set_title("Sobel X")

sobely=Sobel(originalImage,threshold,'y')
axes[index,2].imshow(sobely)
axes[index,2].set_title("Sobel Y")

sobelxy=Sobel(originalImage,threshold,'xy')
axes[index,3].imshow(sobelxy)
axes[index,3].set_title("Sobel Magnitude")

sobeldir=Sobel(originalImage,threshold,'dir')
axes[index,4].imshow(sobeldir)
axes[index,4].set_title("Sobel Direction")


# ### Step 8- Combination

# In[39]:


# Step 8- Combining Different ColorSpaces and Sobel Variants

def combineEverything(warpedImage, color_threshold, sobel_threshold):
    
    s_channel = ExtractChannel(warpedImage,cv2.COLOR_RGB2HLS,color_threshold,2)
    l_channel = ExtractChannel(warpedImage,cv2.COLOR_RGB2HLS,color_threshold,1)
    y_channel= ExtractChannel(warpedImage,cv2.COLOR_RGB2YUV,color_threshold,0)
    
    sobelx = Sobel(warpedImage, sobel_threshold, 'x')
    sobeldir= Sobel(warpedImage, [0.7,25], 'dir')
    #sobelxy=Sobel(warpedImage, sobel_threshold, 'xy')
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(((s_channel == 1) & (l_channel==1)) & (y_channel==1)) | (sobelx == 1)  ] = 1
    return combined_binary


# In[40]:


#testing on test_images
f, axes= plt.subplots(8,2,figsize=(15,30))

for index,warped in enumerate(warpedImages):
    combinedImage=combineEverything(warped,[100,255],[10,150])
    axes[index,0].imshow(warped)
    axes[index,0].set_title("Warped Image")
    axes[index,1].imshow(combinedImage,cmap='gray')
    axes[index,1].set_title("Combined")



