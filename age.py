# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:44:37 2019

@author: user
"""

import colorsys
import dlib                     #人脸识别的库dlib             #数据处理的库numpy
import cv2   
import random                   #图像处理的库OpenCv
from PIL import Image
import PIL.ImageOps 
import numpy as np    


class face_emotion():
    age =0;

    def __init__(self):
      
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor('/home/pi/face/model/shape_predictor_68_face_landmarks.dat')

        #建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置视频参数，propId设置的视频参数，value设置的参数值
        self.cap.set(3, 480)
        # 截图screenshoot的计数器
        self.cnt = 0
        
    def learning_face(self):
        # cap.isOpened（） 返回true/false 检查初始化是否成功
        while(self.cap.isOpened()):

            flag, im_rd = self.cap.read()

            # 每帧数据延时1ms，延时为0读取的是静态帧
            k = cv2.waitKey(10)


           
            font = cv2.FONT_HERSHEY_SIMPLEX
          
            im_rd = cv2.putText(im_rd, "S: screenshot", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "Q: quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(im_rd, (int(200), int(100)), (int(400), int(400)), (255, 255, 255),2)
            
            #im_rd = cv2.putText(im_rd, age,(20,350),font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            # 按下s键截图保存
            cv2.imshow("camera", im_rd)
            if (k == ord('s')):
                self.cnt+=1
                cv2.imwrite("screenshoot"+str(self.cnt)+".jpg",im_rd)
                im1 = cv2.imread("screenshoot"+str(self.cnt)+".jpg")
                w,h,ret=im1.shape   #img.shape可以获得图像的形状，返回值是一个包含行数，列数，通道数的元组
                im2 = np.zeros((w,h,3),np.uint8)

                grey=np.zeros((w,h,3),np.uint8)#生成像素空数组，整数型。
                im2=cv2.cvtColor(im1,cv2.COLOR_BGR2YCrCb)
                lower_range = np.array([0, 133, 77])
                upper_range = np.array([256, 173, 127])
                grey=cv2.inRange(im2, lower_range, upper_range)

                output = cv2.bitwise_and(im1, im1, mask = grey)
                cv2.imwrite("./output.jpeg",output)
                img = cv2.imread("./output.jpeg")

                img = cv2.imread("screenshoot"+str(self.cnt)+".jpg", 0)
          
                dets = self.detector(img, 1)

                for k, d in enumerate(dets):

                 shape=self.predictor(img,d)
                 bwidth=shape.part(62).x-shape.part(61).x 
                 
                 
                 im = Image.open("./output.jpeg")
                
                 
                 region =  im.crop((shape.part(48).x, shape.part(8).y+bwidth,shape.part(54).x,shape.part(8).y+bwidth*5))
                 region1 =  im.crop((shape.part(18).x, shape.part(29).y,shape.part(39).x,shape.part(48).y-bwidth))
                 region2 =  im.crop((shape.part(42).x, shape.part(29).y,shape.part(25).x,shape.part(48).y-bwidth))
                 
                 region.save("./crop_test.jpeg")
                 region1.save("./crop_test1.jpeg")
                 region2.save("./crop_test2.jpeg")
                 
                 
                 
                 img1 = cv2.imread("./crop_test.jpeg")
                 img2 = cv2.imread("./crop_test1.jpeg")
                 img3 = cv2.imread("./crop_test2.jpeg")
                 
                 img4 = cv2.GaussianBlur(img1,(3,3),0)
                 img5 = cv2.GaussianBlur(img2,(3,3),0)
                 img6 = cv2.GaussianBlur(img3,(3,3),0)
                 
                 canny = cv2.Canny(img4, 50, 120)
                 canny1 = cv2.Canny(img5,50,120)
                 canny2 = cv2.Canny(img6,50,120)
                 area = 0
                 point = 0
                 ret1,th1 = cv2.threshold(canny,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                 ret2,th2 = cv2.threshold(canny1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                 ret3,th3 = cv2.threshold(canny2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                 cv2.imshow('th1', th1 )
                 cv2.imshow('th2', th2 )
                 cv2.imshow('th3', th3 )
                 cv2.imshow('canny', canny )
                 cv2.imshow('canny1', canny1 )
                 cv2.imshow('canny2', canny2 )
        
                 cv2.imshow('img4', img4 )
                 cv2.imshow('img5', img5 )
                 cv2.imshow('img6', img6 )
        
                 height, width = th1.shape
                 height1, width1 = th2.shape
                 height2, width2 = th3.shape
        
                for i in range(height):
                  for j in range(width):
                   if th1[i, j] == 255:
                    point  += 1
                for i in range(height1):
                  for j in range(width1):
                   if th2[i, j] == 255:
                    area +=1
                for i in range(height2):
                  for j in range(width2):
                   if th3[i, j] == 255:
                    area+=1
                if (area<15):
                     if(point<=20):
                         age = int(random.randint(20,23)) 
                     else:
                         age = int(random.randint(25,30))
                    
                         
                elif(area<=35):
                     
                     if(point<=5):
                         age = int(random.randint(25,30))
                     else:
                         age = int(random.randint(30,40))
                    
                         
                elif(area<=100 ):
                    
                     if(point<=20):
                         age = int(random.randint(30,40))
                    
                     elif(point<=100):
                         age = int(random.randint(40,50))
                     else:
                         age = int(random.randint(50,60))
                         
                elif(area<=200):
                    if(point<=30):
                         age = int(random.randint(40,50))
                    elif(point<=200):
                         age = int(random.randint(50,60))
                     
                    else:
                         age = int(random.randint(60,65))
                         
                else:
                     if(point<=120):
                         age = int(random.randint(50,60))
                
                     else:
                         age = int(random.randint(60,65))
                
                print("age is:"+repr(age))
                print(area)
                print(point)
                cv2.waitKey(0)
                cv2.destroyAllWindows()      

                
    
            # 按下q键退出
            if(k == ord('q')):
                break

            # 窗口显示
         

        # 释放摄像头


        # 删除建立的窗口
        cv2.destroyAllWindows()

if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()    
