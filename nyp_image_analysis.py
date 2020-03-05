#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##膠化不良分析


# In[554]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageStat
from matplotlib.ticker import FuncFormatter


temp = []
temp_n = []

def changey(temp, position):
    return int(temp/2)

for i in range(80,85):
    i=i+1
    pth = 'test_img/test_ng_image/'+str(i)+'.jpg'
    img_n = cv2.imread(pth)
    img_n = cv2.cvtColor(img_n,cv2.COLOR_BGR2RGB)
    img_n = cv2.cvtColor(img_n,cv2.COLOR_RGB2GRAY)
    img_n = img_n.flatten()
    temp_n.append(img_n)
    
    img = cv2.imread(pth)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img.flatten()[j] > 5:
                temp.append(img.flatten()[j])


# In[ ]:


##灰階分布統計(附帶有無遮罩比較)


# In[555]:



plt.figure(figsize=(15,8))
'''''
plt.subplot(2,2,1)
plt.title('No_good_people without mask(five pics)')
plt.xlabel('Gray_value')
plt.ylabel('Count')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
plt.hist(temp_n,bins=5)
'''''
plt.subplot(2,2,2)
plt.title('No_good_people mask(five pics)')
plt.xlabel('Gray_value')
plt.ylabel('Count')
plt.hist(temp,bins=5,normed=True)
plt.ylim(0,0.015)
#plt.bar(range(1,257),hist)

plt.savefig('ng_gray_people')
plt.show()


# In[ ]:


##批次讀NG圖並計算影像平均亮度與標準差


# In[550]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageStat

mean = []
std = []
mean_n = []
std_n = []
temp = []

for i in range(1,6):
    i = i+1
    pth = 'test_img/test_ng_image/'+str(i)+'.jpg'
    im = Image.open(pth).convert('L')
    pix = im.load()
    
    m_n = np.mean(im)  
    mean_n.append(m_n)
    s_n = np.std(im)
    std_n.append(s_n)
    
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if pix[i,j] > 5:
                temp.append(pix[i,j])
    
    
    #stat = ImageStat.Stat(temp)
    m = np.mean(temp)  
    mean.append(m)
    s = np.std(temp)
    std.append(s)
    #print('平均亮度:',mean)
    #print('標準差:',std)
plt.figure(figsize=(10,8))
'''''
plt.subplot(2,2,1)
plt.title('No_good luminance STD without mask(five pics)')
plt.xlabel('STD')
plt.ylabel('Count')
#plt.ylim((0,20))
plt.xlim((60,90))
plt.hist(std_n)
#plt.savefig('std_ng')
'''''
plt.subplot(2,2,2)
plt.title('No_good luminance STD mask(five pics)')
plt.xlabel('STD')
plt.ylabel('Count')
#plt.xlim(45,61)
plt.ylim(0,20)
plt.hist(std,normed=True)
plt.savefig('std_ng')
plt.show()


# In[ ]:


##RGB分析(G通道)


# In[238]:



temp = []

for i in range(80,81):
    i=i+1
    pth = 'test_img/test_ng_image/'+str(i)+'.jpg'
    img_rgb = cv2.imread(pth)
    (B,G,R) = cv2.split(img_rgb)
   # img_rgb = cv2.imread('test_img/test_ng_image/1.jpg')
    #color = ('b','g','r')
    for i in range(img_rgb.shape[1]):
        for j in range(img_rgb.shape[0]):
            if G.flatten()[j] > 5:
                temp.append(G.flatten()[j])

    #for i ,col in enumerate(color):
     #   hist = cv2.calcHist([img_rgb],[i],None,[200],[50,150])
        
        
       # plt.plot(hist,color=col)
        #plt.xlim([50,150])
plt.figure(figsize=(15,10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

plt.subplot(2,2,1)
plt.title('Green channel with mask')
plt.xlabel('Gray value')
plt.ylabel('Count')
plt.hist(temp)
'''
plt.subplot(2,2,2)
plt.title('Green channel without mask')
plt.xlabel('Gray value')
plt.ylabel('Count')
plt.hist(G.flatten())
'''''
plt.savefig('rgb_ng')
plt.show()


# In[ ]:


##Texture Analysis (entropy,dissimilarity,contrast)


# In[540]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image 
from skimage.feature import greycomatrix,greycoprops
from sklearn.metrics.cluster import entropy

temp_n = []
temp1_n = []
temp2_n = []

temp = []
temp1 = []
temp2 = []

for i in range(80,85):
    i = i+1
    pth = 'test_img/test_ng_image/'+str(i)+'.jpg'
    #img_n = cv2.imread(pth)
    #img_n = cv2.cvtColor(img_n,cv2.COLOR_BGR2RGB)
    #img_n = cv2.cvtColor(img_n,cv2.COLOR_RGB2GRAY)
   
    
    im = Image.open(pth)
    #img = cv2.imread(pth)
    greyImg = np.array(im.convert('L', colors=8))
    #img = cv2.imread(pth)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    
    
    for i in range(greyImg.shape[1]):
        for j in range(greyImg.shape[0]):
            if greyImg[j][i] > 5:
               greyImg[j][i] = greyImg[j][i] 
               
                


    #glcm_m_n = greycomatrix(img_n, distances=[1], angles=[0, np.pi / 4, np.pi / 2],symmetric=True,normed=True)
    #temp1_n.append(greycoprops(glcm_m_n,'dissimilarity')[0,0]) 
    #print(temp1_n)
    #temp2_n.append(greycoprops(glcm_m_n,'contrast')[0,0])
    
    glcm_m = greycomatrix(greyImg, distances=[1], angles=[0, np.pi / 4, np.pi / 2],symmetric=True,normed=True)
    temp1.append(greycoprops(glcm_m,'dissimilarity')[0,0])
    #print(temp1)
    temp2.append(greycoprops(glcm_m,'contrast')[0,0])

#print(glcm_m)
    #enp_n = entropy(img_n)
    #temp_n.append(enp_n)
    
    enp = entropy(img)
    temp.append(enp)
    #print('entropy:',enp)

plt.figure(figsize=(18,10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)
'''
plt.subplot(6,6,1)
plt.title('Entropy without mask')    
plt.xlabel('entropy')
plt.ylabel('count')

plt.hist(temp_n) 

plt.subplot(6,6,2)
plt.title('Dissimilarity without mask')    
plt.xlabel('dissimilarity')
plt.ylabel('count')

plt.hist(temp1_n) 

plt.subplot(6,6,3)
plt.title('Contrast without mask')    
plt.xlabel('contrast')
plt.ylabel('count')

plt.hist(temp2_n) 
'''''
plt.subplot(3,3,1)
plt.title('No_good_people Entropy mask')    
plt.xlabel('entropy')
plt.ylabel('count')

plt.hist(temp,normed=True) 

plt.subplot(3,3,2)
plt.title('No_good_people Dissimilarity mask')    
plt.xlabel('dissimilarity')
plt.ylabel('count')
plt.ylim(0,2)
plt.hist(temp1,normed=True) 

plt.subplot(3,3,3)
plt.title('No_good_people Contrast mask')    
plt.xlabel('contrast')
plt.ylabel('count')
plt.ylim(0,0.35)

plt.hist(temp2,normed=True) 
plt.savefig('entropy_ng')



# In[ ]:


##Texture Analysis (LBP)


# In[403]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from skimage.feature import local_binary_pattern
from matplotlib.ticker import FuncFormatter

temp = []
temp_n = []
r = 1
point = 8*r

def changey(temp, position):
    return int(temp/250)

for i in range(1,6):
    i = i+1
    pth = 'test_img/test_ng_image/'+str(i)+'.jpg'
    img = cv2.imread(pth)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #mask = img > 0
    LBP = local_binary_pattern(img,point,r)    
    temp_n.append(LBP.flatten())
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if LBP.flatten()[j] > 5:
                temp.append(LBP.flatten()[j])
#cv2.imwrite('out.jpg',img)    

#print(temp)
#eps = 1e-7
#(hist,_) = np.histogram(temp)

#hist = hist.astype("float")

#hist /= (hist.sum() + eps)
#print(hist)
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.title("No_good local Binary Patterns mask")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel class")
#plt.ylim(0,70)
#plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
plt.hist(temp,normed=True)
plt.savefig('lbp_ng')
'''''
plt.subplot(2,2,2)
plt.title("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel class")
plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
#plt.savefig('lbp_ng')
plt.hist(temp_n)    
#plt.imshow(LBP,cmap='gray') 
'''''
plt.show()


# In[ ]:


##膠化良好分析


# In[542]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageStat
from matplotlib.ticker import FuncFormatter


temp = []
temp_n = []

def changey(temp, position):
    return int(temp/2)

for i in range(1,6):
    i=i+1
    pth = 'test_img/test_g_image/'+str(i)+'.jpg'
    img_n = cv2.imread(pth)
    img_n = cv2.cvtColor(img_n,cv2.COLOR_BGR2RGB)
    img_n = cv2.cvtColor(img_n,cv2.COLOR_RGB2GRAY)
    
    img_n = img_n.flatten()
    temp_n.append(img_n)
    
    img = cv2.imread(pth)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    

    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img.flatten()[j] > 5:
                temp.append(img.flatten()[j])


# In[ ]:


##灰階分布統計


# In[544]:


plt.figure(figsize=(15,8))
'''''
plt.subplot(2,2,1)
plt.title('Without mask(five pics)')
plt.xlabel('Gray_value')
plt.ylabel('Count')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
plt.hist(temp_n,bins=5)
'''''
plt.subplot(2,2,2)
plt.title('Good mask(five pics)')
plt.xlabel('Gray_value')
plt.ylabel('Count')
plt.ylim(0,0.015)
plt.hist(temp,bins=5,normed=True)

#plt.bar(range(1,257),hist)

plt.savefig('g_gray')
plt.show()


# In[ ]:


##批次讀G圖並計算影像平均亮度與標準差


# In[551]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageStat

mean = []
std = []
mean_n = []
std_n = []
temp = []

for i in range(1,6):
    i = i+1
    pth = 'test_img/test_g_image/'+str(i)+'.jpg'
    im = Image.open(pth).convert('L')
    pix = im.load()
    
    m_n = np.mean(im)  
    mean_n.append(m_n)
    s_n = np.std(im)
    std_n.append(s_n)
    
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if pix[i,j] > 5:
                temp.append(pix[i,j])
    
    
    #stat = ImageStat.Stat(temp)
    m = np.mean(temp)  
    mean.append(m)
    s = np.std(temp)
    std.append(s)
    #print('平均亮度:',mean)
    #print('標準差:',std)
plt.figure(figsize=(10,8))
'''''
plt.subplot(2,2,1)
plt.title('Luminance STD without mask(five pics)')
plt.xlabel('STD')
plt.ylabel('Count')
#plt.ylim((0,20))
plt.xlim((60,90))
plt.hist(std_n)
#plt.savefig('std_ng')
'''''
plt.subplot(2,2,2)
plt.title('Good luminance STD mask(five pics)')
plt.xlabel('STD')
plt.ylabel('Count')
plt.ylim(0,20)
plt.hist(std,normed=True)
plt.savefig('std_g')
plt.show()


# In[ ]:


##RGB分析(G通道)


# In[46]:


temp = []

for i in range(80,85):
    i=i+1
    pth = 'test_img/test_g_image/'+str(i)+'.jpg'
    img_rgb = cv2.imread(pth)
    (B,G,R) = cv2.split(img_rgb)
   # img_rgb = cv2.imread('test_img/test_ng_image/1.jpg')
    #color = ('b','g','r')
    for i in range(img_rgb.shape[1]):
        for j in range(img_rgb.shape[0]):
            if G.flatten()[j] > 5:
                temp.append(G.flatten()[j])

    #for i ,col in enumerate(color):
     #   hist = cv2.calcHist([img_rgb],[i],None,[200],[50,150])
        
        
       # plt.plot(hist,color=col)
        #plt.xlim([50,150])
plt.figure(figsize=(15,10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

plt.subplot(2,2,1)
plt.title('Green channel with mask')
plt.xlabel('Gray value')
plt.ylabel('Count')
plt.hist(temp)

plt.subplot(2,2,2)
plt.title('Green channel without mask')
plt.xlabel('Gray value')
plt.ylabel('Count')
plt.hist(G.flatten())
#plt.savefig('rgb_ng')
plt.show()


# In[ ]:


##Texture Analysis (entropy,dissimilarity,contrast)


# In[541]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

import pandas as pd
from PIL import Image ,ImageDraw
from skimage.feature import greycomatrix,greycoprops
from sklearn.metrics.cluster import entropy

temp_n = []
temp1_n = []
temp2_n = []

temp = []
temp1 = []
temp2 = []
#img1 = Image.new( "RGB", (340,141) )

for i in range(1,6):
    i = i+1
    pth = 'test_img/test_g_image/'+str(i)+'.jpg'
    #img_n = cv2.imread(pth)
    #img_n = cv2.cvtColor(img_n,cv2.COLOR_BGR2RGB)
    #img_n = cv2.cvtColor(img_n,cv2.COLOR_RGB2GRAY)
   
    
    im = Image.open(pth)
    #img = cv2.imread(pth)
    greyImg = np.array(im.convert('L', colors=8))
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    
    
    for i in range(greyImg.shape[1]):
        for j in range(greyImg.shape[0]):
            if greyImg[j][i] > 5:
               greyImg[j][i] = greyImg[j][i] 
               #print(greyImg)
               
                


    #glcm_m_n = greycomatrix(img_n, distances=[1], angles=[0, np.pi / 4, np.pi / 2],symmetric=True,normed=True)
    #temp1_n.append(greycoprops(glcm_m_n,'dissimilarity')[0,0]) 
    #print(temp1_n)
    #temp2_n.append(greycoprops(glcm_m_n,'contrast')[0,0])
    
    glcm_m = greycomatrix(greyImg, distances=[1], angles=[0, np.pi / 4, np.pi / 2],symmetric=True,normed=True)
    temp1.append(greycoprops(glcm_m,'dissimilarity')[0,0])
    #print(temp1)
    temp2.append(greycoprops(glcm_m,'contrast')[0,0])

#print(glcm_m)
    #enp_n = entropy(img_n)
    #temp_n.append(enp_n)
    
    enp = entropy(img)
    temp.append(enp)
    #print('entropy:',enp)

plt.figure(figsize=(18,10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)
'''
plt.subplot(6,6,1)
plt.title('Entropy without mask')    
plt.xlabel('entropy')
plt.ylabel('count')

plt.hist(temp_n) 

plt.subplot(6,6,2)
plt.title('Dissimilarity without mask')    
plt.xlabel('dissimilarity')
plt.ylabel('count')

plt.hist(temp1_n) 

plt.subplot(6,6,3)
plt.title('Contrast without mask')    
plt.xlabel('contrast')
plt.ylabel('count')

plt.hist(temp2_n) 
'''''
plt.subplot(3,3,1)
plt.title('Good Entropy mask')    
plt.xlabel('entropy')
plt.ylabel('count')

plt.hist(temp,normed=True) 

plt.subplot(3,3,2)
plt.title('Good Dissimilarity mask')    
plt.xlabel('dissimilarity')
plt.ylabel('count')
plt.ylim(0,2)

plt.hist(temp1,normed=True) 

plt.subplot(3,3,3)
plt.title('Good Contrast mask')    
plt.xlabel('contrast')
plt.ylabel('count')
plt.ylim(0,0.35)
plt.hist(temp2,normed=True) 
plt.savefig('entropy_g')


# In[ ]:





# In[446]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from skimage.feature import local_binary_pattern
from matplotlib.ticker import FuncFormatter

temp = []
temp_n = []
r = 1
point = 8*r

def changey(temp, position):
    return int(temp/250)

for i in range(1,6):
    i = i+1
    pth = 'test_img/test_g_image/'+str(i)+'.jpg'
    img = cv2.imread(pth)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #mask = img > 0
    LBP = local_binary_pattern(img,point,r)    
    temp_n.append(LBP.flatten())
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if LBP.flatten()[j] > 5:
                temp.append(LBP.flatten()[j])
#cv2.imwrite('out.jpg',img)    

#print(temp)
#eps = 1e-7
#(hist,_) = np.histogram(temp)

#hist = hist.astype("float")

#hist /= (hist.sum() + eps)
#print(hist)
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.title("Good local Binary Patterns mask")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel class")
#plt.ylim([0,0.05])
#plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
plt.hist(temp,normed=True)
plt.savefig('lbp_g')
'''''
plt.subplot(2,2,2)
plt.title("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel class")
plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
#plt.savefig('lbp_ng')
plt.hist(temp_n)    
#plt.imshow(LBP,cmap='gray') 
'''''
plt.show()



# In[ ]:




