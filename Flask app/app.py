import numpy as np
import time
import os
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image




from flask import Flask , request, render_template, Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("weed.h5")

@app.route('/photo',methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.form["action"]
        if(f=="photo"):
            return render_template('base.html')
        elif(f=="video"):
            return render_template('video.html')
        elif(f=="live"):
            return render_template('live.html')
@app.route('/')
def options():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict_classes(x)
        4
            
        print("prediction",preds)
            
        index = ['Broadleaf','Grass','Soil','Soybean']
        
        text = str(index[preds[0]])
        
    return text



def uploadVideo(filepath):
    
        
        video = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        name = ["Broadleaf","Grass","Soil","Soybean"]

        while(video.isOpened()):
    
            success, frame = video.read()
            if success==True:
                cv2.imwrite("image.jpg",frame)
                img = image.load_img("image.jpg",target_size = (64,64))
                x = image.img_to_array(img)
                x = np.expand_dims(x,axis=0)
                pred = model.predict_classes(x)
                p = pred[0]
                print(pred)
                cv2.putText(frame, "Predicted crop = "+ str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                #print(frame.shape)
                #cv2.imshow("image",frame)
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                if(cv2.waitKey(1) & 0xFF == ord('a')):
                    break
            else:
                break
       


@app.route('/video1',methods=['GET','POST'])
def video():
    if request.method == 'POST':
        f = request.files['video']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        return Response(uploadVideo(filepath),mimetype='multipart/x-mixed-replace; boundary=frame')
 



def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    name = ["Broadleaf","Grass","Soil","Soybean"]
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite("image.jpg",frame)
            img = image.load_img("image.jpg",target_size = (64,64))
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis=0)
            pred = model.predict_classes(x)
            p = pred[0]
            print(pred)
            cv2.putText(frame, "Predicted crop = "+ str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            frame = cv2.resize(frame, (0,0), fx=2, fy=1.5) 
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break
        

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')       

if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    