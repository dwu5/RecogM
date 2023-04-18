import multiprocessing

import cv2
from utils import camera_predict, emotion_classifier

model = emotion_classifier()
# graph = tf.get_default_graph()

# detector_path = 'E:\\Anaconda3\\envs\\bishe\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
detector_path = 'E:\\Anaconda3\\envs\\bishe\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml'

# 开启摄像头
cap = cv2.VideoCapture(0)
# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def detect_predict(cap=cap):
    # 在每一帧数据中进行人脸识别
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 以灰度图的形式读取图像cv2.COLOR_RGB2GRAY (cv2.COLOR_BGRA2BGR)

            # 实例化OpenCV人脸识别的分类器
            face_detector = cv2.CascadeClassifier(detector_path)

            # 调用识别人脸
            faceRects = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 框出人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(camera_predict(img[y:y + h, x:x + w], model))

            cv2.imshow("frame", frame)
            # print(faceRects)

            # 点击小写字母q 退出程序
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 点击窗口关闭按钮退出程序
            if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1.0:
                break


# detect_predict(cap)
cap_process = multiprocessing.Process(target=detect_predict)
cap_process.run()

# # 在每一帧数据中进行人脸识别
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 以灰度图的形式读取图像cv2.COLOR_RGB2GRAY (cv2.COLOR_BGRA2BGR)
#
#         # 实例化OpenCV人脸识别的分类器
#         face_detector = cv2.CascadeClassifier(detector_path)
#
#         # 调用识别人脸
#         faceRects = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
#         for faceRect in faceRects:
#             x, y, w, h = faceRect
#
#             # 框出人脸
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             print(camera_predict(img[y:y+h, x:x+w], model))
#
#         cv2.imshow("frame", frame)
#         # print(faceRects)
#
#         # 点击小写字母q 退出程序
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         # 点击窗口关闭按钮退出程序
#         if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1.0:
#             break

# 释放资源
cap.release()
cv2.destroyAllWindows()

# rect = classifier.detectMultiScale(img, scaleFactor, minNeighbors, minSize,maxsize)
# img: 要进行检测的人脸图像
# scaleFactor: 前后两次扫描中，搜索窗口的比例系数
# minNeighbors：目标至少被检测到minNeighbors次才会被认为是目标
# minSize和maxSize: 目标的最小尺寸和最大尺寸
# https://blog.51cto.com/u_13977270/3395806
