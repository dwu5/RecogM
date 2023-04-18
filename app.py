import base64
from io import BytesIO

from PIL import Image

from flask import Flask, render_template, request, Response
from werkzeug.datastructures import FileStorage

from utils import *

app = Flask(__name__)
# 导入模型
model = emotion_classifier()


# graph = tf.get_default_graph()


# @app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        img_fs = request.files['file']  # 缓存取出使用后，无法再次读取
        # <FileStorage: 'bike.jpg' ('image/jpeg')>
        filename = img_fs.filename
        print('image: {}'.format(filename))

        img_bytes = img_fs.read()  # bytes
        save_img(img_bytes, filename)

        img = base64.b64encode(img_bytes).decode()  # str

        # 执行预测
        # img_input = Image.open('upload/{}'.format(filename))
        img_input = f"upload\\{filename}"
        if img_input:
            pred, show_viz = emotion_predict(img_input, model)
        else:
            pred, show_viz = "null", "null"
        print(pred)

        return render_template('result.html', img=img, pred=pred, show_viz=show_viz)


@app.route('/camera', methods=['GET'])
def camera():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(get_frame(model), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
