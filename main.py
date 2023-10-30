from ultralytics import YOLO
import cv2
import imutils
import numpy as np
import pytesseract
import re
import threading

def scanLisence(ScanImage):
    print("scan edilir:")
    gray = cv2.cvtColor(ScanImage,cv2.COLOR_BGR2GRAY)  #convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)#Blur to reduce noise
            #gray=cv2.morphologyEx(gray,cv2.MORPH_RECT,(3,3))
            #gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    edged = cv2.Canny(gray, 30, 500)  #Perform Edge detection
    #cv2.imshow("BB",edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
            
    Text=None
    for c in cnts:
        ScreenCnt = None
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        applen=len(approx)
        if True:
            ScreenCnt = approx
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(
            mask,
            [ScreenCnt],
            0,
            255,
            -1,
            )
            new_image = cv2.bitwise_and(ScanImage, ScanImage, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            #cv2.imshow("Frame",gray)
          
                        #reader = easyocr.Reader(['en'], gpu = True) 
                        #print(reader.readtext(Cropped))
            Text = pytesseract.image_to_string(Cropped,config='--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                       
            Text=re.sub('[^A-Za-z0-9]+', '', Text)
            if Text!="" and Text is not None and len(Text)>6:
                           #cv2.drawContours(ScanImage, [ScreenCnt], -1, (0, 255, 0), 3)
                print(Text)
                            
            else:
                Text=None




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
        results = model.track(frame, persist=True)

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

            if(conf>0.6):
                x1=cords[0]
                y1=cords[1]
                x2=cords[2]
                y2=cords[3]

                croppedImage=frame[y1:y2, x1:x2]
            
                threading.Thread(target=scanLisence,args=(croppedImage,)).start()
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break