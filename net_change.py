import re
import urllib2
import urllib
from bs4 import BeautifulSoup
import cookielib
from pytesser import *
import Image
import numpy as np
'''for i in range(100):
    #req=urllib2.Request(url='https://sfrz.cqupt.edu.cn:8443/cas/captcha.htm')
    im_data=urllib2.urlopen('https://sfrz.cqupt.edu.cn:8443/cas/captcha.htm').read()
    f=open('image'+str(i)+'.jpeg','wb')
    #f=open('imag1e.jpeg','w')
    f.write(im_data)
    f.close()'''

def convert_to_bw(im):
    im=im.convert("L")
    im=im.point(lambda x: 255 if x > 196 else 0)
    im.convert("1")
    return im

def split(im,name):
    
    result=[]
    w,h=im.size
    xs=[-1,17,35,53,71]
    ys=[0,27]
    for i,x in enumerate(xs):
        if i+1>=len(xs):
            break
        box=(xs[i]+1,ys[0],xs[i+1],28)
        t=im.crop(box).copy()
        #filename=name+str(i)+'.bmp'
        t.save('sample\\'+name+str(i)+'.bmp')
def normallise_32_32(im,width,high):
    pass

def process_victor(im):
    im_p=im.load()
    n=0
    num=[]
    #print im_p[21,23]+im_p[1,2]
    for j in range(0,32,2):
        for i in range(0,32,2):
            num.append(4-(im_p[i,j]+im_p[i+1,j]+im_p[i,j+1]+im_p[i+1,j+1])/255)
            n=n+1
    return num



def read_pic():
    result=[]
    data=[]

    for i in (range(10)):
        res_vic=[0 for x in range(10)]
        res_vic[i]=1

        for j in range(1,5):

            im=Image.open('sample\\'+str(i)+str(j)+'.bmp')
            im=im.convert('1')
            #w,h=im.size
            #data_raw=list(im.getdata())
            im_resize=im.resize((32,32))
            #im_resize.show()
            victor=process_victor(im_resize)
            #print len(data)
            #print data_raw
            
            #data_af_nor=normallise_32_32(data_raw,w,h)
            result.append([victor,res_vic])
    return result   











def writetxt(data):
    fp=open('result.txt','r+')
    for x in result:
        for i in x:
            fp.write("%s "%str(i))

            
        fp.write('\n')


result=read_pic()




'''for i,filen in [(10,'10'),(11,'11')]:
    im=Image.open('rain\image%s.jpeg'%str(i))
    im=convert_to_bw(im)
    split(im,filen)'''



'''-----------------------------'''
def fuc(data):
    return 1/(1+(np.exp(data*(-1))))

def net(data):
    learningRate = 0.4
    desiredError = 0.001
    maxIterations = 10000

    value_ih=(np.random.rand(256,64)-0.5)*2
    #print value_ih.shape
    value_ho=(np.random.rand(64,10)-0.5)*2     # creat init value
    samples=np.array(result)
    #for iteration in range(maxIterations):


    samples_input=samples[:,0]
    samples_result=samples[:,1]
    samples_num=samples.shape[0]
    e=np.zeros(samples_num)
    E=0.0
    #print samples_result.shape
    for iteration in range(maxIterations):
        for k in range(samples_num):
            #while 1:
            sample=np.array(samples_input[k]).reshape(np.array(samples_input[k]).shape[0],1).astype(np.float64)
            #sample.astype(np.float64)  #  256X1
            sample_max=np.max(sample)
            sample_min=np.min(sample)
            
            sample=(sample-sample_min)/(sample_max-sample_min)
            #print sample.dtype
            
            Des=np.array(samples_result[k]).reshape(np.array(samples_result[k]).shape[0],1)   #    10x1
            #print sample.shape
            hiden_input=sample.T.dot(value_ih)
            #print hiden_input.dtype  
            #break
            
            #break          
            hiden_input=hiden_input.reshape(hiden_input.shape[1],1)    #  64X1

            #print 'hiden_input.shape--> ',hiden_input.shape
            hiden_output=fuc(hiden_input)
            #print hiden_output.shape
            Y_in=hiden_output.T.dot(value_ho)    #  1X10
            #print Y_in.shape
            #break       
            Y_out=fuc(Y_in)

            e[k]=np.sum(np.square(np.subtract(Y_out,Des.T)))
            if e[k]>desiredError:
                d_f_Yin=fuc(Y_in)
                sigma_out=(Des.T-Y_out)*(d_f_Yin*(1-d_f_Yin))     #  1x10
                #print sigma_out
                d_f_hi=fuc(hiden_input) 
                #d_f_hi=d_f_hi.reshape(1,d_f_hi.shape[0])
                #break
                sigma_hiden=(sigma_out.dot(value_ho.T))*(d_f_hi*(1-d_f_hi))   # 1X64
                #print sigma_hiden
                #break
                reverse_value_ho=learningRate*(hiden_output).dot(sigma_out)
                #print reverse_value_ho
                reverse_value_ih=learningRate*(sample.dot(sigma_hiden))

                value_ho=value_ho+reverse_value_ho
                value_ih=value_ih+reverse_value_ih
                #else :
                    #break

            #print Y_out
            #print e[k]
            #break
        #print Y_out
        #print e[k]
        #np.savez('net_value.npz',a=value_ih,b=value_ho)
        break    
        E=np.sum(e)
        if E<desiredError:
            print '-------learn complete!!----'
            print iteration
            break
        if (iteration%100)==0:
            print 'report time-->' ,iteration ,'error-->',E
    
net(result)
