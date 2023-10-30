import cv2
import numpy as np
import pytesseract
import imutils

import imutils
import numpy as np
import re

def scanLisence(ScanImage):
    #cv2.imshow("S",ScanImage)
    print("Oxuma basladi")
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


path='car/'
yolo_cfg = path+'yolov3.cfg' # YOLO modelinin yapılandırma dosyası
yolo_weights = path+'yolov3.weights' # YOLO modelinin ağırlık dosyası
yolo_classes = path+'coco.names' # Sınıf adlarının bulunduğu dosya

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# YOLO modelinin sınıflarını yükleyin
classes = []
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Ana resmi yükleyin
ana_resim = cv2.imread('images/az.jpeg')
height, width, _ = ana_resim.shape

# YOLO için giriş görüntüsünü hazırlayın
blob = cv2.dnn.blobFromImage(ana_resim, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# YOLO ile nesne algılama yapın
net.setInput(blob)
layer_names = net.getUnconnectedOutLayersNames()
outs = net.forward(layer_names)

# Algılanan nesneleri işaretleyin
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5  # Algılama için güven eşiği
nms_threshold = 0.4  # Non-maximum suppression eşiği

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression uygula
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Algılanan nesneleri işaretleyin
for i in indices:
   # i = i[0]  # İndeks öğesini al
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # İşaretlemeyi ve etiketi çizin
    cv2.rectangle(ana_resim, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(ana_resim, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    scanLisence(ana_resim[y-10:y+h+10, x-10:x+w+10])

# Sonucu göster
cv2.imshow('Nesne Algılama', ana_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
