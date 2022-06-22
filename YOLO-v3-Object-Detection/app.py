from email.mime import image
from flask import Flask, jsonify, request
from yolo_detection_images import detectObjects
from werkzeug.utils import secure_filename
import os,json,pickle,time
from flask_restful import Api, Resource
import multiprocessing as mp
from save import detectSave
#from json_result import getJsonProcess

app=Flask(__name__)
api = Api(app)  # Flask 객체에 Api 객체 등록

image_folder = '/home/g2019sun0925/flask_yolo_api/YOLO-v3-Object-Detection/images'
json_folder = '/home/g2019sun0925/flask_yolo_api/YOLO-v3-Object-Detection/json'

@app.route('/photo', methods=['GET', 'POST'])
def photo_to_json():
    #start_time = time.time_ns()
    f = request.files['file']
    f.save(os.path.join(image_folder, secure_filename(f.filename)))
    image_abs_path = os.path.join(image_folder, secure_filename(f.filename))
    json_path=os.path.join(json_folder, secure_filename(f.filename))
    process = mp.Process(target=detectSave, args=(image_abs_path, json_path))
    process.start()
    return "ok"


class getJson(Resource):
    def get(self, filename):
        print("getJson start")
        json_abs_path=os.path.join(json_folder, filename)
        if(os.path.isfile(json_abs_path)):
            print("파일있음")
            with open(json_abs_path, 'rb') as fr:
                result = pickle.load(fr)
                #os.remove(image_abs_path)
        else:
            print("파일없음")
            result = {'No object detected':'Nothing'}
        print(result)
        return jsonify(result)

api.add_resource(getJson, '/getJson/<string:filename>')

app.run(host = '0.0.0.0', port = 5000)
