'''
This file is part of the public domain software distributed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 
'''

import numpy as np


class Canny():
    '''
    This class contains all the necessary functions for detecting edges using the Canny edge detector algorithm implemented from scratch using nothing but numpy.
    '''

    def pad(self,img,pad):
        '''This Function add the padding to the input image, this function uses same padding([(n + 2p) x (n + 2p) image] * [(f x f) filter] —> [(n x n) image])
        parameters: 
        image -> input image that needs to be padded
        pad -> number of pads to be added
        '''
        

        #Create an array([(n + 2p) x (n + 2p)]) of zeros 
        padded = np.zeros((img.shape[0]+pad*2,img.shape[1]+pad*2))
        
        #Insert image in the center of the image of the array of zeros
        padded[pad:-pad,pad:-pad] = img

        #return the padded image and the numer of pads
        return padded



    def conv2D(self,img,filter, padding=True):
        '''This function will perform a 2D convolve on the given image with the given filter
        parameters:
        img -> image to be convolved
        filter -> filter to be applied to the image
        to_pad -> boolean indicating whether to pad or not the image
        '''

        # Add Padding in order to get the input sized output image 
        #Find the number of pixels needed to pad
        p = (filter.shape[0]-1)//2

        #Check if padding is required
        if padding == True:
            padded = self.pad(img,p)
        else:
            padded = img

        #Create the output matrix
        conv = np.zeros((img.shape[0],img.shape[1]))


        #Loop in the matrix for the size of the image
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):

                # Convolve only if the y moved appropriate strides
                if j%p==0: #Stride

                    #Convolution 
                    # G(i,j) = sum(filter * Image(x,y))
                    conv[i,j] = (filter* padded[i-1:i+2, j-1:j+2]).sum()


        # Return the convolved matrix
        return conv

    def GaussianBlur(self,img, fsize=3):
        '''This function will produce the gaussian blur of the given image
        parameters:
        img -> Image that needs to be smoothed using Gaussian Blur
        filter -> Gaussian filter size, default is 3x3
        '''

        # Create a Gaussian filter of the given kernel size
        filter = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]])
        # Convolve the image with the gaussian filter to the gaussian blur
        return self.conv2D(img,filter)





    def Sobel(self,img, dx=1,dy=1,fsize=3):
        '''This function will produce the first order derivative using Sobel filter and the gradient direction of the given image
        parameters:
        img -> Image that needs to be derviated using Sobel filter
        dx -> if dx==1: derivative with respect to x(vertical), default value is 1
        dy -> if dy==1: derivative with respect to y(horizontal), default value is 1
        fsize -> Filter size, default is 3x3
        '''

        #Defing Sobel Filters
        filter_x = np.array(((-1,0,1),(-2,0,2),(-1,0,1)))
        filter_y = filter_x.transpose()
        
        #img , p = pad(img, filter_x)

        #Creating convolved image matrix
        conv = np.zeros(img.shape)

        # Creating f derivative w.r.t x and y matrix
        dx = np.zeros(img.shape)
        dy = np.zeros(img.shape)
        
        if dx==1:
            dx = self.conv2D(img,filter_x,False)
        if dy==1:
            dy = self.conv2D(img,filter_y,False)

        #Calculating the magnitude of the gradients
        conv = np.sqrt(np.square(dx)+np.square(dy))
        conv *= 255/np.max(conv)

        #np.arctan2(dy, dx) Computes element-wise arc tangent of dx/dy choosing the quadrant appropriately
        #Converting radians to degree --> (-180,180)
        direction = np.rad2deg(np.arctan2(dy, dx))  
        #Adding 180 to get the range to (0,180)
        direction[direction <0] += 180
        #conv = conv.astype('uint8')

        return conv, direction


    def Laplacian(self,img,fsize=3):
        '''This function calculates the second derivative of the gradients, that is Laplacian of Gaussian in this case
        parameters:
        img -> Image that needs to be derivativated with Laplacian of Gaussian Filter
        fsize -> Filter size, default is 3x3
        '''
        #Defining 3x3 Laplacian of Gaussian filter
        filter = np.array(((0,1,0),(1,-4,1),(0,1,0)))

        #Convolving the image with the filter
        conv = self.conv2D(img,filter,False)

        #Normalizing the resulting image
        conv /= np.max(conv)
        return conv

    
    def non_maxima_suppression(self,img, direction):
        '''This function supresses the non maximum values to 0
        parameters:
        img -> Image that needs non maximum values to be supressed
        direction -> gradient direction matrix
        '''

        supressed = np.zeros(img.shape)

        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                p = 255
                r = 255

                #Angle 0
                if (0<= direction[i,j]<22.5)or(157.5 <= direction[i,j]<=180):
                    p = img[i,j-1]
                    r = img[i,j+1]

                #Angle 45
                elif (22.5<=direction[i,j]<67.5):
                    p = img[i-1,j+1]
                    r = img[i+1,j-1]

                #Angle 90
                elif(67.5<=direction[i,j]<112.5):
                    p = img[i-1,j]
                    r = img[i+1,j]

                #Angle 135
                elif(112.5<=direction[i,j]<157.5):
                    p = img[i-1,j-1]
                    r = img[i+1,j+1]

                #Check if the pixel is larger than the adjacent pixels
                if (img[i,j]>=p and img[i,j]>=r):
                    supressed[i,j] = img[i,j]
    
                else:
                    supressed[i,j] = 0
                    
        # Scalling the pixels to the maximum value
        supressed *= 255.0/supressed.max()
        return supressed


    def hystersis(self,img, low, high, weak=50,strong=255):
        '''
        This function will use double thresholding to differentiate true edges and then perfoms edge tracking using hystersis.
        parameters:
        img -> Input image for tracking edges
        low -> Lower threshold
        high -> Upper threshold
        weak -> pixel value that is considered to be weak, default value is 50
        strong -> pixel value that is considered to be strong, default value is 255
        '''

        #Creating new matrix of image size with zeros
        thresh_out = np.zeros(img.shape)

        #Finding the index where the pixel is higher than the high threshold
        strong_i, strong_j = np.where(img >= high)

        ##Finding the index where the pixel is lower than the low threshold
        weak_i, weak_j = np.where((img <= high) & (img >low))
        
        '''Assinging the values 255 and weak(i.e., given or 50 by default) to 
        indexes with strong values and weak values respectively'''
        thresh_out[strong_i,strong_j] = strong 
        thresh_out[weak_i,weak_j] = weak


        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
            
                try:
                    #Checking if the pixel is between the high and low threshold
                    if (thresh_out[i,j] == weak):
                        #If the value is between the high and low threshold, then check if the neighboring pixel is an edge
                        if (thresh_out[i-1,j-1]==strong or thresh_out[i-1,j]==strong or thresh_out[i-1,j+1]==strong or thresh_out[i,j-1]==strong or thresh_out[i,j+1]==strong or thresh_out[i+1,j-1]==strong or thresh_out[i+1,j]==strong or thresh_out[i+1,j+1]==strong):
                            #If the neighboring pixel is an edge, then make this pixel an edge
                            thresh_out[i,j]=strong 
                except IndexError as e:
                    pass

        #push all pixel values to 0 that are not an edge
        thresh_out[thresh_out!=strong] = 0

        
        return thresh_out


    def detect_edge(self, img, low, high):
        '''
        This function detects the edge using the Canny edge detection algorithm.
        parameters:
        img -> Image to detect edges
        low -> lower threshold value for hystersis
        high -> upper threshold value for hystersis
        '''
        gaussian = self.GaussianBlur(img)
        sobel, direction = self.Sobel(gaussian)
        laplacian = self.Laplacian(gaussian)
        nms = self.non_maxima_suppression(sobel, direction)
        hystersis = self.hystersis(nms,low,high)
        return  hystersis
        