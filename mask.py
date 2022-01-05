#-*- coding:utf-8 -*-
#@Time :  22:53
#@Author: Thomas
#@File :mask.py
#@Software : PyCharm

import paddlehub as hub
import cv2

module = hub.Module(name="pyramidbox_lite_mobile_mask")     #口罩检测模型
face_cascade = cv2.CascadeClassifier("/Users/tomwu/PycharmProjects/mask_test/haarcascade_frontalface_default.xml")     #人脸识别分类器


frame=cv2.imread("/Users/tomwu/Downloads/meidai.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

if len(faces) > 0:
    for faceRect in faces:
        x, y, w, h = faceRect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h // 2, x:x + w]
        roi_color = frame[y:y + h // 2, x:x + w]
        cv2.imwrite("/Users/tomwu/Downloads/yes.jpeg",roi_gray)

cv2.imshow('frame', frame)      #展现
file_name="/Users/tomwu/Downloads/daile_hui.jpg"       #图片存储路径
cv2.imwrite(file_name, frame)       #写入获取的一帧
#口罩检测模型
input_dict = {"data": [cv2.imread(file_name)]}
results = module.face_detection(data=input_dict)
#判断是否检测到人脸和是否戴口罩
mask = results[0]["data"]
if len(mask) != 0:
    mask_test = mask[0]['label']
    if mask_test=="NO MASK":
        print('没戴口罩')
    else:
        print('戴口罩了')
else:
    print("未检测到人脸！请将脸移入摄像头视角范围内！")

cv2.imwrite(file_name,frame)