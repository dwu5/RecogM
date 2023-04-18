import os
import base64
from io import BytesIO

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from matplotlib import pyplot as plt

# emotion_labels = ["Angry", "Disgusted", "Fear", "Happy", "Sad", "Surprised", "Neutral"]
emotion_labels = ["Angry", "Disgusted", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

# model_path = 'E:\\bishe\\models\\XCEPTION_epoch23_acc0.61.h5'
# model_path = 'E:\\bishe\\models\\1227\\model_1227.h5'
# model_path = 'E:\\bishe\\models\\vgg\\vgg19_0115.h5'   # 摄像头：neutral, surprised, happy, angry， sad
# model_path = 'E:\\bishe\\models\\resnet\\resnet50_0131.h5'
# model_path = 'E:\\bishe\\models\\vgg\\vgg19_0206.h5'  # 摄像头最佳之一
# model_path = 'E:\\bishe\\models\\vgg\\vgg19_rafdb_20230301\\50-0.84518.hdf5'   # 摄像头可用之一
model_path = 'models/vgg19_rafdb_50-0.84518.hdf5'

# detector_path = 'E:\\Anaconda3\\envs\\bishe\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'
# detector_path = 'E:\\Anaconda3\\envs\\bishe\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
detector_path = 'models/faceDetector/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(detector_path)


def emotion_classifier():
    print("loading...")
    model = load_model(model_path)
    # model_json_file = 'models/model.json'
    # with open(model_json_file, "r") as json_file:
    #     config = json_file.read()
    #     model = tf.keras.models.model_from_json(config)
    print("model loaded")
    return model


# 数据归一化
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


"""
    四通道，三通道，单通道
    三维，二维
"""


# def img_format(img):
    # xception (48,48,1)
    # img_48 = img.resize((48, 48), Image.ANTIALIAS)  # 调整分辨率
    # img_gray = img_48.convert('L')  # 转为灰度图
    #
    # img_np = np.asarray(img_gray)  # 转为np.array
    # img_np = preprocess_input(img_np)  # 数据归一化
    # img_np = np.expand_dims(np.expand_dims(img_np, axis=0), axis=3)  # 调整向量维度
    # return img_np

    # model_1227 (48,48,1)
    # img = cv2.imread(img)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_48 = cv2.resize(img_gray, (48, 48))
    # img_input = img_48.reshape(1, 48, 48, 1)

    # vgg (48,48,3)
    # img = cv2.imread(img)
    # img_48 = cv2.resize(img, (48, 48))
    # img_input = img_48.reshape(1, 48, 48, 3)

    # vgg (100,100,3), raf-db
    # img = cv2.imread(img)
    # img_100 = cv2.resize(img, (10, 100))
    # img_input = img_100.reshape(1, 100, 100, 3)

    # return img_input


def face_detect(img):
    img = cv2.imread(img)
    face = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=10)
    x, y, w, h = 0, 0, 0, 0
    for (x, y, w, h) in face:
        x, y, w, h = x, y, w, h
    if x == 0 and w == 0:
        print("No face detected!")
        face_48 = cv2.resize(img, (48, 48))
        face_100 = cv2.resize(img, (100,100))
    else:
        face_48 = cv2.resize(img[y:y + h, x:x + w], (48, 48))
        face_100 = cv2.resize(img[y:y + h, x:x + w], (100,100))
    # face_data = face_48.reshape(1, 48, 48, 3)
    face_data = face_100.reshape(1, 100, 100, 3)
    return face_data


def emotion_predict(input, model):
    # img_input = img_format(input)
    # print(f"img_input: {img_input.shape}")
    # with graph.as_default():
    #     pred = model.predict(img_input)

    # pred = model.predict(img_input)
    pred = model.predict(face_detect(input))

    data = pd.DataFrame(pred[0], index=emotion_labels, columns=['rate'])
    viz = sns.barplot(x=emotion_labels, y='rate', data=data)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
    data = base64.encodebytes(buffer.getvalue()).decode()
    show_viz = 'data:image/png;base64,' + str(data)
    # 关闭，否则画出来的图是重复的
    plt.close()

    return emotion_labels[np.argmax(pred)], show_viz


def camera_predict(input, model):
    # img48 = cv2.resize(input, (48, 48))
    # img_input = img48.reshape(1, 48, 48, 3)

    img100 = cv2.resize(input, (100, 100))
    img_input = img100.reshape(1, 100, 100, 3)

    # with graph.as_default():
    #     pred = model.predict(img_input)
    pred = model.predict(img_input)
    return emotion_labels[np.argmax(pred)]


def save_img(img_bytes, filename):
    # path = 'static/'
    path = 'upload/'
    if not os.path.exists(path):
        os.mkdir(path)

    img_pil = Image.open(BytesIO(img_bytes))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    img_pil.save(path + filename)


def get_frame(model):
    # 开启摄像头
    camera = cv2.VideoCapture(0)
    # 设置分辨率
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 在每一帧数据中进行人脸识别
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 以灰度图的形式读取图像cv2.COLOR_RGB2GRAY (cv2.COLOR_BGRA2BGR)
            # 调用识别人脸
            faceRects = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=10)
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框出人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(camera_predict(img[y:y + h, x:x + w], model))
                cv2.putText(frame, camera_predict(img[y:y + h, x:x + w], model), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # 把获取到的图像格式转换(编码)成流数据，赋值到内存缓存中;
            # 主要用于图像数据格式的压缩，方便网络传输
            ret1, buffer = cv2.imencode('.jpg', frame)
            # 将缓存里的流数据转成字节流
            frame = buffer.tobytes()
            # 指定字节流类型image/jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
