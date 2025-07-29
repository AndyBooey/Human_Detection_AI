import cv2  #OpenCV, to handle image processing
import numpy as np  #to handle arrays for image data
import time
from picamera2 import Picamera2 #efficient way of accessing Raspberry Pi Camera
import tflite_runtime.interpreter as tflite #tensorflow lite model
import smtplib  #email APi to send email alerts
from email.message import EmailMessage

#pip3 install tflite-runtime --break-system-packages
#https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip


# Load model and labels
interpreter = tflite.Interpreter(model_path="detect.tflite")    # "detect.tflite" is an AI model used to recognize objects
interpreter.allocate_tensors()  #loads the AI model into memory for later use
input_details = interpreter.get_input_details()     #gets info about model's input tensor (shape, dtype, and index) for later use for cropping image for input
output_details = interpreter.get_output_details()   #gets info about model's output tensor (bounding bodes, class IDs, and confidence scores)

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
    input_data = np.expand_dims(input_data, axis=0)    #crop (batch dimension) to make it fit the model's input size
    input_data = np.uint8(input_data)   #ensure image is in uint8 format

    interpreter.set_tensor(input_details[0]['index'], input_data)   #load the image into the model's input tensor
    interpreter.invoke()    #runs the model

    #THIS GETS THE PREDICTION OF THE CLASS IN LABELMAP.TXT (RETURNS AN INDEX)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]      # Bounding box coords
    classes = interpreter.get_tensor(output_details[1]['index'])[0]    # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]     # Confidence scores
    #LATER IT GRABS THE INDEX WITH THE BEST CONFIDENCE SCORE AND OUTPUTS THE LABEL

    return boxes, classes, scores

def send_msg():
    sender_email = "personidentifier5@gmail.com"
    sender_password = "uohywgaeytupktph"
    receiver_email = "tylertran395@gmail.com"
    msg = EmailMessage()
    msg.set_content("This is an automated alert: Theres a person nearby")
    msg['Subject'] = "Python Email Alert!"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Email was sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")


last_alert_time = 0
cooldown_seconds = 10
try:    
    while True:     #starts an INFINITELOOP so that it keeps on recording and capturing frames for object detection
        frame = picam2.capture_array()
        boxes, classes, scores = detect_objects(frame)

        for i in range(len(scores)):
            if labels[int(classes[i])] == "person" and scores[i] > 0.5:     #for each score check if it is above 50%
            

                h, w, _ = frame.shape
                ymin, xmin, ymax, xmax = boxes[i]
                (x1, y1, x2, y2) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))         #all of these convert coordinates to pixel values

                label = labels[int(classes[i])] #this finds the readable label name for the class


                current_time = time.time()
                if current_time - last_alert_time > cooldown_seconds:
                    send_msg()
                    last_alert_time = current_time
        


                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)    #draws the box around the image
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):   #breaks if user presses q
            break

except KeyboardInterrupt:   #breaks if user presses CTRL+C
    picam2.stop()
    cv2.destroyAllWindows()
