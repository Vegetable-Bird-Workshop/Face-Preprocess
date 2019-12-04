#----------------------- 2019.12.4-------------------
#      python:3.6  dlib:19.8.1

import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

frames_save_path = r'C:\Users\user\Desktop\code\Face_preprocess' #存截取的视频图像的文件夹的地址
videos_src_path = r'C:\Users\user\Desktop\lab\AFEW4\Train'      #视频的源地址

key_detect_model_path = r"D:\chorme_download\shape_predictor_68_face_landmarks.dat"

#需要处理的视频的类型
video_formats = [".avi"]
width = 320  #输出图像的长和宽
height = 240
time_interval = 1 #视频采样间隔

def Imageget(videos_src_path, frames_save_path, formats, frame_width, frame_height, interval):


    face_save_path = frames_save_path + '/' + 'prototype_face'
    aligment_face_save_path = frames_save_path + '/' + 'aligment_face'
    frames_save_path = frames_save_path + '/' + 'prototype_pic'
    os.mkdir(frames_save_path)          #创建文件夹存取原图像
    os.mkdir(face_save_path)            #创建文件夹存取人脸原图像
    os.mkdir(aligment_face_save_path)   #存取人脸对齐图像

    def filter_format(x, all_formats):
            if x[-4:] in all_formats:
                return True
            else:
                return False

    emotion_types = os.listdir(videos_src_path)

    detector = dlib.get_frontal_face_detector()
    keydictor = dlib.shape_predictor(key_detect_model_path)

    for types in emotion_types:
        videos = os.listdir(videos_src_path + '/' + types)
        videos = filter(lambda x: filter_format(x, formats), videos)
        os.mkdir(frames_save_path + '/' + types)
        os.mkdir(face_save_path + '/' + types)
        os.mkdir(aligment_face_save_path + '/' + types)
        for each_video in videos:
            print("正在读取视频：", each_video)
    
            each_video_name = each_video[:-4]

            os.mkdir(frames_save_path + '/' + types + '/' + each_video_name)
            os.mkdir(face_save_path + '/' + types + '/' + each_video_name)
            os.mkdir(aligment_face_save_path + '/' + types + '/' + each_video_name)

            each_video_save_full_path = os.path.join(frames_save_path, types, each_video_name) + "/"
            each_face_save_full_path = os.path.join(face_save_path, types, each_video_name) + "/"
            each_aligment_face_save_path = os.path.join(aligment_face_save_path, types, each_video_name) + "/"
            each_video_full_path = os.path.join(videos_src_path, types, each_video)

            cap = cv2.VideoCapture(each_video_full_path)
            frame_index = 0
            frame_count = 0
            
            if cap.isOpened():
                success = True
            else:
                success = False
                print("读取失败!")

            while(success):
                success, frame = cap.read()
                if success:
                    print("---> 正在读取第%d帧:" % frame_index, success)
                    if frame_index % interval == 0:
                        # resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, frame)
                        # boxes_c, landmarks = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        faces = detector(frameRGB,1)

                        for (i, face) in enumerate(faces):
                            (x, y, w, h) = face_coordinate(face)
                            key_pot = keydictor(frameRGB, face)
                            face_aligment = cv2.cvtColor(dlib.get_face_chip(frameRGB, key_pot), cv2.COLOR_BGR2RGB)

                            crop, flag = frame_cut(frame, (x, y, w, h))
                            if flag != False:
                                cv2.imwrite(each_face_save_full_path + "%d_" % frame_count + "%d.jpg" % i, crop)
                                cv2.imwrite(each_aligment_face_save_path + "%d_" % frame_count + "%d.jpg" % i, face_aligment)
                                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        frame_count += 1
                    frame_index += 1

            cap.release()

#dlib的坐标会返回一些负值，需要处理
def frame_cut(frame, location):
    if location[0] > frame.shape[1] or location[1] > frame.shape[0]:
        flag = False
    else:
        flag = True
    return frame[location[1] if location[1]>0 else 0 : 
                 location[1] + location[3] if location[1]+location[3]<frame.shape[0] else frame.shape[0],
                 location[0] if location[0]>0 else 0 : 
                 location[0] + location[2] if location[0]+location[2]<frame.shape[1] else frame.shape[1]], flag

def face_coordinate(face): #获得人脸坐标
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"): # 将shape类的输出转化为numpy数组
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


if __name__ == "__main__":
    Imageget(videos_src_path, frames_save_path, video_formats, width, height, time_interval)