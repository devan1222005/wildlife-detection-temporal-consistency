import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from ultralytics import YOLO

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
model = YOLO("best.pt")  # Keep your trained model here

CONF_THRESHOLD = 0.5
CONSECUTIVE_FRAMES = 5


# ==============================
# Home Route
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        video = request.files["video"]

        if video.filename == "":
            flash("Please upload a video file!")
            return redirect(request.url)

        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        # ==============================
        # BASE YOLO DETECTION
        # ==============================
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        base_detected = False
        temporal_detected = False
        detection_counter = 0
        detected_image_path = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            detections = results[0].boxes

            # --------------------------
            # Base YOLO (Single Frame)
            # --------------------------
            if len(detections) > 0 and not base_detected:
                base_detected = True

            # --------------------------
            # Temporal Consistency
            # --------------------------
            if len(detections) > 0:
                detection_counter += 1
            else:
                detection_counter = 0

            if detection_counter >= CONSECUTIVE_FRAMES:
                temporal_detected = True

                annotated_frame = results[0].plot()
                detected_image_path = os.path.join(DETECTED_FOLDER, "detected.jpg")
                cv2.imwrite(detected_image_path, annotated_frame)
                break

        cap.release()

        base_result = "Detected" if base_detected else "Not Detected"
        temporal_result = "Confirmed" if temporal_detected else "Rejected"

        return render_template(
            "index2.html",
            base_result=base_result,
            temporal_result=temporal_result,
            image="detected/detected.jpg" if temporal_detected else None
        )

    return render_template("index2.html")


if __name__ == "__main__":
    app.run(debug=True)
