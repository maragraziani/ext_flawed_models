
# coding: utf-8

# In[4]:


import sys
sys.path.append('intentionally_flawed_models/')
from dataset_utils import ImageNet10Random 
import models
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
import os
import sys
import h5py 


# In[5]:


import dataset_utils
reload(dataset_utils)
from dataset_utils import ImageNet10Random 


# In[6]:


classes = classes=['n02085620', # chiuaua 1075
         'n02099601', # golden retriever 967
         'n02165456', # ladybug 1574 
         'n02676566', # acoustic guitar 1083
         'n02701002', # ambulance 249 --- TO CHANGE Knitwear
         'n02871525', # bookshop 1050 
         'n02927161', # butcher 1026
         'n03000134', # chainlink fence 1239
         'n03042490', # cliff dwelling 1335
         'n03089624', # confectionery 
            ]


# In[7]:


PATH='/mnt/nas2/results/IntermediateResults/Mara/SDLCV'
imgnet05=ImageNet10Random(classes=classes,  
                          label_corrupt_p=0.5, 
                          path_to_train='{}/imagenet_data/data-train.h5'.format(PATH), 
                          path_to_val='{}/imagenet_data/data-val.h5'.format(PATH))


# In[8]:


reload(models)
from models import CNN


# In[10]:


model = CNN(deep=2,save_fold='/mnt/nas2/results/IntermediateResults/Mara/probes/imagenet/2H_lcp0.5')


# In[11]:


model.save_fold


# In[12]:


model.model.summary()


# In[13]:


model.train_and_monitor_with_rcvs(imgnet05,[5,9], custom_epochs=10)

