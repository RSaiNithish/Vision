'''
This file is part of the public domain software distributed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 
'''

import numpy as np

class Convolution:
 
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
