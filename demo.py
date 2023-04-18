from utils import preprocess_input

emotion_labels = ["Angry", "Disgusted", "Fear", "Happy", "Sad", "Surprised", "Neutral"]
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from keras.models import load_model

# 载入模型
model = load_model('D:\\bishe\\model\\XCEPTION_epoch23_acc0.61.h5')
dataset_path = 'D:\\bishe\\dataset\\fer2013.csv'  # 文件保存位置
image_size = (48, 48)  # 图片大小

# test
img = Image.open('upload/angry1.jpg')
img_np = np.asarray(img)
img_np = preprocess_input(img_np)
img_test = np.expand_dims(np.expand_dims(img_np, axis=0), axis=3)
pred = model.predict(img_test)

data = pd.DataFrame(pred[0], index=emotion_labels, columns=['rate'])
graph = sns.barplot(x=emotion_labels, y='rate', data=data)

print(emotion_labels[np.argmax(pred)])
print(graph)

# img = cv2.imread('upload/cyc.png')
# print(type(img))
# img_48 = cv2.resize(img, (48, 48))
# print(img_48.shape)
