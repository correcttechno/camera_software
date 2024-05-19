import cv2
from openalpr import Alpr
import Jetson.GPIO as GPIO
import time


GPIO.setmode(GPIO.BOARD)
output_pin = 29
GPIO.setup(output_pin, GPIO.OUT)

plates=["10LL841","99ZF822","99ZR822"]

def main():
    # OpenALPR nesnesini başlat
    alpr = Alpr("sg", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")

    # Alpr başlatılamazsa hata mesajı ver
    if not alpr.is_loaded():
        print("OpenALPR yüklenemedi. Lütfen yüklemeyi kontrol edin.")
        return

    # USB kamerayı başlat
    cap = cv2.VideoCapture(0)
  
    while True:
        # Görüntüyü al
        ret, frame = cap.read()
        
        # Plakaları tanı
        results = alpr.recognize_ndarray(frame)

        # Tanımlanan plakaları çerçevele
      
        for plate in results['results']:
            for candidate in plate['candidates'][:1]:
                # Plakayı ve güven puanını al
                plate_str = candidate['plate']
                confidence = candidate['confidence']
                

                if plate_str in plates:
                    GPIO.output(output_pin, GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(output_pin, GPIO.LOW)
                print(plate_str)


                # Plakayı ve güven puanını çerçevele
                if confidence > 0:
                    coordinates = plate['coordinates']
                    cv2.rectangle(frame, (coordinates[0]['x'], coordinates[0]['y']), (coordinates[2]['x'], coordinates[2]['y']), (255, 0, 0), 2)
                    cv2.putText(frame, plate_str , (coordinates[0]['x'], coordinates[0]['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Çerçeveli görüntüyü göster
        #cv2.imshow('OpenALPR', frame)

        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Temizlik ve çıkış
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == '__main__':
    main()
