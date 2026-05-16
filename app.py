import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage

# ==============================
# Flask Setup
# ==============================
app = Flask(__name__)
app.secret_key = "wildanimal_secret"

UPLOAD_FOLDER = "static/uploads"
DETECTED_FOLDER = "static/detected"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# ==============================
# Load YOLO Model
# ==============================
model = YOLO("best.pt")

CONF_THRESHOLD = 0.5
CONSECUTIVE_FRAMES = 5


# ==============================                                                                                                        
# Email Alert Function
# ==============================
def send_email_alert(image_path, animal_name):

    sender_email = "EMAIL_USER"
    sender_password = "EMAIL_PASS"
    receiver_email = "RECEIVER_EMAIL"

    msg = EmailMessage()
    msg["Subject"] = f"🚨 {animal_name} Detected in Farm!"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(
        f"""
Wild Animal Alert!

Detected Animal : {animal_name}

A harmful wild animal has been detected in the farm surveillance system.

Please check the attached frame immediately.
"""
    )

    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(image_path)

    msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)


# ==============================
# Main Route
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        video = request.files["video"]

        if video.filename == "":
            flash("Please upload a video file.")
            return redirect(request.url)

        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        cap = cv2.VideoCapture(video_path)

        detection_counter = 0
        detected_animal = None
        detected_image_path = None

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            detections = results[0].boxes

            if len(detections) > 0:

                detection_counter += 1

                class_id = int(detections.cls[0])
                detected_animal = model.names[class_id]

            else:
                detection_counter = 0

            if detection_counter >= CONSECUTIVE_FRAMES:

                annotated_frame = results[0].plot()

                detected_image_path = os.path.join(DETECTED_FOLDER, "detected.jpg")
                cv2.imwrite(detected_image_path, annotated_frame)

                send_email_alert(detected_image_path, detected_animal)

                break

        cap.release()

        if detected_image_path:

            flash(f"🚨 {detected_animal} detected! Email alert sent.")

            return render_template(
                "index.html",
                image="detected/detected.jpg",
                animal=detected_animal
            )

        else:

            flash("✅ No wild animal detected.")
            return redirect(url_for("index"))

    return render_template("index.html")


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=False)