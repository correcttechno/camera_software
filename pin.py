import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
output_pin = 29
GPIO.setup(output_pin, GPIO.OUT)

try:
    while True:
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
