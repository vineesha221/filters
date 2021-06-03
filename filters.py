import cv2
import numpy as np
img = cv2.imread(r'C:\Users\vineesha thoutam\Downloads\song.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rows,cols = img.shape[:2]



cv2.imshow('Original',img)

#identity
kernel_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
output=cv2.filter2D(img,-1,kernel_identity)
cv2.imshow('Identity filter',output)

#blur
kernel_3x3 = np.ones((3,3),np.float32) / 9.0
output=cv2.filter2D(img,-1,kernel_3x3) 
cv2.imshow('Identity filter',output)

#larger blur
kernel_5x5 = np.ones((5,5),np.float32) / 25.0
output=cv2.filter2D(img,-1,kernel_5x5)
cv2.imshow('Identity filter',output)

#motion blur
size=15
kernel_motion_blur = np.zeros((size,size))
kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur = kernel_motion_blur /  size
output=cv2.filter2D(img,-1,kernel_motion_blur)
cv2.imshow('Motion blur',output)

#sharpen
kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
output1=cv2.filter2D(img,-1,kernel_sharpen_1)
cv2.imshow('Sharpening',output1)

#more shapening
kernel_sharpen_2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
output2=cv2.filter2D(img,-1,kernel_sharpen_2)
cv2.imshow('Excessive sharpening',output2)

#edge enhancement
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
output3=cv2.filter2D(img,-1,kernel_sharpen_3)
cv2.imshow('Edge Enhancement',output3)

#emboss
kernel_emboss1=np.array([[0,-1,1],[1,0,-1],[1,1,0]])
output=cv2.filter2D(gray,-1,kernel_emboss1)+128
cv2.imshow('Emboss1',output)

#emboss2
kernel_emboss2=np.array([[-1,-1,0],[-1,0,11],[0,1,1]])
output=cv2.filter2D(gray,-1,kernel_emboss2)+128
cv2.imshow('Emboss2',output)

#emboss3
kernel_emboss3=np.array([[1,0,0],[0,0,0],[0,0,-1]])
output=cv2.filter2D(gray,-1,kernel_emboss3)+128
cv2.imshow('Emboss3',output)

#Sobel
sobel_horizontal = cv2.Sobel(img,cv2.CV_64F, 1,0,ksize=5)
cv2.imshow('Sobel horizontal',sobel_horizontal)

#Sobel2
sobel_vertical = cv2.Sobel(img,cv2.CV_64F, 1,0,ksize=5)
cv2.imshow('Sobel vertical',sobel_vertical)

#erode
kernel_erode=np.ones((5,5),np.uint8)
img_erosion = cv2.erode(img,kernel_erode,iterations = 1)
cv2.imshow('Erode',img_erosion)

#dilate
img_dilatation = cv2.dilate(img,kernel_erode,iterations = 1)
cv2.imshow('Dilate',img_dilatation)

#vignette
kernel_gauss_x= cv2.getGaussianKernel(cols,200)
kernel_gauss_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_gauss_y * kernel_gauss_x.T
mask=255*kernel/np.linalg.norm(kernel)
output=np.copy(img)
for i in range(3):
  output[:,:,i]=output[:,:,i] * mask
cv2.imshow('Vignette',output)

#shifted vignette
kernel_gauss_x= cv2.getGaussianKernel(int(1.5*cols),200)
kernel_gauss_y = cv2.getGaussianKernel(int(1.5*rows),200)
kernel = kernel_gauss_y * kernel_gauss_x.T
mask=255*kernel/np.linalg.norm(kernel)
mask = mask[int(0.5*rows):,int(0.5*cols):]
output=np.copy(img)
for i in range(3):
  output[:,:,i]=output[:,:,i] * mask
cv2.imshow('Shifted Vignette',output)

#contrast:RGB
img_yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_yuv[:,:,0]=cv2.equalizeHist(img_yuv[:,:,0])
output=cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('Histogram equalized',img)

cv2.waitKey(0)