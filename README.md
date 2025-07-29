## 👀 Human Detection with Email Alerts
This project uses a pre-trained MobileNet SSD v1 TensorFlow Lite model to detect humans in real-time using a Raspberry Pi and PiCamera.
📧 When a person is detected with 50% confidence or higher, the system automatically sends an email alert to a specified address.


 ##🛠 Features
- ✅ Real-time human detection on Raspberry Pi

- 🎯 50% confidence threshold for alerts

- 📧 Email notifications via Gmail SMTP

- 📷 Bounding box and label display on the live camera feed

- 🔍 Testing mode available (no email alerts)




## 📂 Project Files
## human_alert_cam.py
- detects humans in live camera feed and sends email alerts
- If a person is detected with over 50% confidence, it:
  - Draws a bounding box around the person.
  - Labels the detection with the class name "person"
  - Sends an email alert with a cooldown interval to prevent spamming.
- Integrates OpenCV for video display and TensorFlow for inference
- OpenCV is used for video display
- TensorFlow Lite is used for inference (the confidence score)
- Uses Gmail SMTP to send alerts

## preview_detector.py
- Simpler version of human_alert_cam.py designed for testing and debugging
- Performs live footage of person detection but does not send email alerts.

## detect.tflite
- This is a pre-trained MobileNet SSD model commonly used with devices like the Raspberry Pi.
- Used for object detection on the footage of a live camera.
- Used to detect 90 object categories, including "person".

## lablemap.txt
- Contains a list of 90 class labels corresponding to the model's output class IDs.
- Each line corresponds to a class ID expected in the model's output.
- Includes "person," the label used to trigger email alerts.


## 🚀 How to Run
1️⃣ Prerequisites
- Raspberry Pi with Python 3 installed
- PiCamera connected and enabled

- Required libraries installed:
- pip install opencv-python tensorflow tensorflow-lite smtplib

2️⃣ Running Human Detection with Email Alerts
> 🖥️ **Run Command:**
> ```bash
> python human_alert_cam.py
> ```


3️⃣ Running in Preview Mode (No Email Alerts)
> 🖥️ **Run Command:**
> ```bash
python preview_detector.py
> ```




