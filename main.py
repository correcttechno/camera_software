from ultralytics import YOLO
import cv2
import numpy as np
import threading
import easyocr



reader = easyocr.Reader(['en'], gpu=False)


def scanLisence(croppedImage):
    license_plate_crop_gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)
    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
    detections = reader.readtext(license_plate_crop_thresh)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        print(text)
        return text
    return ''

# load yolov8 model
model = YOLO('car/first.pt')

# load video
cap = cv2.VideoCapture(0)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True,verbose=False)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()
        result=results[0]
        
        if(len(result.boxes)>0):
            box=result.boxes[0]
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            """print("Object type:", class_id)
            print("Coordinates:", cords)
            print("Probability:", conf)
            print("---")"""

            if(conf>0.1):
                x1=cords[0]
                y1=cords[1]
                x2=cords[2]
                y2=cords[3]

                croppedImage=frame[y1:y2, x1:x2]
                cv2.imshow('CROPPED', croppedImage)
                
               
                #croppedImage()
                threading.Thread(target=scanLisence,args=(croppedImage,)).start()
                
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                

        # visualize
        cv2.imshow('frame', frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break