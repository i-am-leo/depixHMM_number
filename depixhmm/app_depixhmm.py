# coding=utf-8
import os
import sys
from io import BytesIO


from flask import Flask, request, send_file, jsonify
from flask_restful import Api, Resource
from flask_httpauth import HTTPBasicAuth
from text_depixelizer.HMM.depix_hmm import DepixHMM
from PIL import Image, ImageDraw, ImageFont

USER_DATA = {
    "admin": "SuperSecretPwd"
}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
api = Api(app)
auth = HTTPBasicAuth()

import traceback
from datetime import datetime

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

import threading
import queue

queueLock = threading.Lock()
tokens = queue.Queue()
tokens.put("T1")
tokens.put("T2")
tokens.put("T3")
tokens.put("T4")
tokens.put("T5")
tokens.put("T6")

import pickle


@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password


@app.errorhandler(Exception)
def handle_bad_request(error):
    """Catch BadRequest exception globally, serialize into JSON, and respond with 400."""
    payload = dict()
    try:
        traceback.print_exc()
        payload['message'] = repr(error)
    finally:
        return jsonify(payload), 400


class MosaicsTextClean(Resource):

    @auth.login_required
    def post(self):
        # obtain token
        token = tokens.get(block=True, timeout=20)

        media_path = None
        result_file = None
        try:
            # checking if the file is present or not.
            if 'file' not in request.files:
                raise Exception("file not found")

            # if util.is_img(file):
            file = request.files['file']
            # pixelatedImagePath = os.path.join("../tmp", (datetime.now().strftime("%Y%m%d%H%M%S")) + '.jpg')
            # file.save(pixelatedImagePath)

            font = request.form.get('font')  # 获取字体
            font_size = request.form.get('font_size')  # 获取字体大小
            # print(type(font_size))
            block_size = request.form.get('block_size')  # 获取马赛克格子大小

            # logging.info("加载马赛克图片： %s" % pixelatedImagePath)
            # pixelatedImage = LoadedImage(pixelatedImagePath)

            # 定义状态信息
            state = {
                'font_size': font_size,
                'block_size': block_size
            }
            logging.info("加载模型")
            # with open(f'arial_50_blocksize-10.pkl', 'rb') as f:
            with open(f'{font}_{font_size}_blocksize-{block_size}.pkl', 'rb') as f:
                hmm = pickle.load(f)
                # 调用__setstate__方法进行对象状态的设置
                # hmm.setstate(state, font_size, block_size)
            try:
                print('开始对马赛克图片进行识别')
                with Image.open(file) as img:
                    reconstructed_string: str = hmm.test_image(img)
            except ValueError:
                return "文件大小不符合规范"

            print(reconstructed_string)

            return reconstructed_string

        finally:
            tokens.put(token, block=False)
            tokens.task_done()


if __name__ == '__main__':
    api.add_resource(MosaicsTextClean, "/mosaics-text/clean/")
    app.run(host='0.0.0.0', port=8082, debug=True)
