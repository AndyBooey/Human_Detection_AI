import cv2  #OpenCV, to handle image processing
import numpy as np  #to handle arrays for image data
import time
from picamera2 import Picamera2 #efficient way of accessing Raspberry Pi Camera
import tflite_runtime.interpreter as tflite #tensorflow lite model

#pip3 install tflite-runtime --break-system-packages
#https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip


# Load model and labels
interpreter = tflite.Interpreter(model_path="detect.tflite")    # "detect.tflite" is an AI model used to recognize objects
interpreter.allocate_tensors()  #loads the AI model into memory for later use
input_details = interpreter.get_input_details()     #gets the input details of an image ("if .tflite expects a face, it retrieves the face from image as an input")
output_details = interpreter.get_output_details()   #gives the coordinates of the image that matches the .tflite model

with open("labelmap.txt", "r") as f:    #this reads the class names within the labelmap.txt (for labeling each object)
    labels = [line.strip() for line in f.readlines()]

# Setup Pi camera
picam2 = Picamera2()    
picam2.preview_configuration.main.size = (320, 240) #runs 320x240 resolution
picam2.preview_configuration.main.format = "RGB888" 
picam2.configure("preview") #open for live preview
picam2.start()

time.sleep(1)   #wait one second for camera to warm up

#function that detects objects for each FRAME
def detect_objects(image): 
    input_data = cv2.resize(image, (300, 300))  #resize image to be like model input size
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # Bounding box coords
    classes = interpreter.get_tensor(output_details[1]['index'])[0]    # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # Confidence scores

    return boxes, classes, scores

try:
    while True:
        frame = picam2.capture_array()
        boxes, classes, scores = detect_objects(frame)

        for i in range(len(scores)):
            if scores[i] > 0.5:
                h, w, _ = frame.shape
                ymin, xmin, ymax, xmax = boxes[i]
                (x1, y1, x2, y2) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
                label = labels[int(classes[i])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    picam2.stop()
    cv2.destroyAllWindows()
