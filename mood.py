import cv2
import dlib
import numpy as np 

# 使用特徵提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib的68點模型，使用作者訓練好的特徵預測器
predictor = dlib.shape_predictor("/home/pi/face/model/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cnt = 0
while(cap.isOpened()):
    line_brow_x = []
    line_brow_y = []
    line_mouth_x = []
    line_mouth_y = []
    flag, im_rd = cap.read()
    k = cv2.waitKey(10)
    font = cv2.FONT_HERSHEY_SIMPLEX
          
    im_rd = cv2.putText(im_rd, "S: screenshot", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "Q: quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.rectangle(im_rd, (int(200), int(100)), (int(400), int(400)), (255, 255, 255),2)
    cv2.imshow("camera", im_rd)
    if (k == ord('s') or k==ord('a')):
        cnt+=1
        cv2.imwrite("screenshoot"+str(cnt)+".jpg",im_rd)
        img = cv2.imread("screenshoot"+str(cnt)+".jpg")
        cv2.imshow("1",img)

        dets = detector(img, 1)
        print("人臉數：", len(dets))
        for k, d in enumerate(dets):
            print("第", k+1, "個人臉d的座標：",
              "left:", d.left(),
              "right:", d.right(),
              "top:", d.top(),
              "bottom:", d.bottom())

            face_width = d.right() - d.left()
            face_heigth = d.bottom() - d.top()

        #print('人臉面積為：',(width*heigth))
                # 利用預測器預測
            shape = predictor(img, d)
        # 標出68個點的位置
            for i in range(68):
                cv2.circle(img, (shape.part(i).x, shape.part(i).y), 4, (0, 255, 0), -1, 8)
                cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        # 顯示一下處理的圖片，然後銷燬視窗
            cv2.imshow('face', img)
            #cv2.waitKey(0)
        # 眉毛
            brow_sum = 0    # 高度之和SSS
            frown_sum = 0   # 兩邊眉毛距離之和
         #嘴巴高度
            mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
            mouth_hight = shape.part(66).y - shape.part(62).y/face_heigth
        #眼睛高度shape.part(62).y
            eye_hight = ( (shape.part(44).y - shape.part(46).y)+(shape.part(37).y - shape.part(41).y) )/2
            for j in range(17, 21):
                            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
                            frown_sum += shape.part(j + 5).x - shape.part(j).x
                            line_brow_x.append(shape.part(j).x)
                            line_brow_y.append(shape.part(j).y)

            tempx = np.array(line_brow_x)
            tempy = np.array(line_brow_y)
            z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
            brow_k =-round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的

            brow_hight = (brow_sum / 10) / face_width  # 眉毛高度占比
            brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
            print("眉毛距离:",round(brow_width,3))
                        
                       
                        # print("眉毛高度与识别框高度之比：",round(brow_arv/self.face_width,3))
                        # print("眉毛间距与识别框高度之比：",round(frown_arv/self.face_width,3))

                        # 眼睛睁开程度
            eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                                   shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
            eye_hight = (eye_sum / 4) / face_width
            nose_width=shape.part(35).x-shape.part(31).x
                        #print("眼睛睁开距离与识别框高度之比：",round(eye_open/self.face_width,3))
     
        # 分情況討論
        # 張嘴，可能是開心或者驚訝
            print("眼睛高度:",eye_hight,
             "嘴巴高度：",mouth_hight,
              "眉毛斜率：",brow_k)
            if round(mouth_width >= 0.32):
                
                if round(mouth_hight >= 0.04):
                    print("amazing")
                else:     
                    if eye_hight >= 0.045:
                        print("angry")
                    else:                  
                        print("happy")

            # 沒有張嘴，可能是正常和生氣
            else:
                if eye_hight <= 0.3:                   
                    print("sad")
                else:
                    if mouth_hight>=0.03:
                        print("disgust")
                    else:
                        print("nature")
            break
       

cv2.destroyALLWindow()
