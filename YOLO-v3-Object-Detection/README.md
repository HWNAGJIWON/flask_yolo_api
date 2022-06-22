# YOLO-v3-Object-Detection API 

참고한 동영상: https://www.youtube.com/watch?v=iT1yJTk77CQ 

### YOLO-v3-Object-Detection 폴더 내부 구조
.  
├── LICENSE  
├── app.py  
├── cfg  
│   └── yolov3.cfg  
├── images  
│   ├── box.jpg  
│   └── pet.jpg  
├── json  
│   ├── box.json  
│   └── pet.json  
├── json_result.py  
├── obj.names  
├── save.py  
├── yolo_detection_images.py  
└── yolov3.weights  

1. images 폴더 : 안드로이드에서 전송한 이미지를 저장하는 폴더
2. json 폴더 : 이미지를 탐색한 json 결과 파일을 저장하는 폴더
3. yolov3.weights : 학습시켰다면 학습한 yolov3.weights로 변경해주세요. 
4. obj.names와 cfg/yolv3.cfg도 직접 학습시켰다면 변경시켜주세요.


### app.py  
- 앱에서 받은 이미지를 저장하고, 결과값을 앱으로 전송하는 메인 함수  

### yolo_detection_images.py  
- 객체 인식 결과를 json 파일로 리턴하는 함수  
  
### save.py
- 객체 인식 결과를 내부 파일에 저장하는 함수  

### 자세한 코드 설명
- 블로그에 추가 예정 https://blog.naver.com/digitalnomad00



### 사용 방법
