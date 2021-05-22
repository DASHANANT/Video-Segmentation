#!/usr/bin/env python
# coding: utf-8

# # After training on pets can it segment people?

# In[4]:


import numpy as np
import cv2
import tensorflow as tf
""" Load the model """
model = tf.keras.models.load_model("unet.h5")

cap=cv2.VideoCapture(0)

while (True):
    ret,frame=cap.read()
    cv2.imshow('original',frame) 
    H, W, _ = frame.shape
    ori_frame = frame
    frame = cv2.resize(frame, (128, 128))    
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    mask = model.predict(frame)[0]    
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (W, H))
    cv2.imshow('segmented',mask) 
    combine_frame = ori_frame * mask
    combine_frame = combine_frame.astype(np.uint8)
    cv2.imshow('original segmented',combine_frame)   
    if cv2.waitKey(1)  == 13:
        break  
        
cap.release() 
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




