import cv2
import numpy as np

# YOLO modelini yükleyin


path='default/'
yolo_cfg = path+'yolov3.cfg' # YOLO modelinin yapılandırma dosyası
yolo_weights = path+'yolov3.weights' # YOLO modelinin ağırlık dosyası
yolo_classes = path+'coco.names' # Sınıf adlarının bulunduğu dosya

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# YOLO modelinin sınıflarını yükleyin
classes = []
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# Ana resmi yükleyin
ana_resim = cv2.imread('images/armud.jpeg')
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
    cv2.rectangle(ana_resim, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(ana_resim, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Sonucu göster
cv2.imshow('Nesne Algılama', ana_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
