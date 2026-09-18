import cv2
import requests
import time
import os
from flask import Flask, Response
from picamera2 import Picamera2
from mtcnn import MTCNN

app = Flask(__name__)

picam2 = Picamera2()
config_cam = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
picam2.configure(config_cam)
picam2.start()

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector = MTCNN()

C_TRIGGER_URL = os.environ.get("C_TRIGGER_URL", "http://localhost:6000/trigger")
COOLDOWN = int(os.environ.get("TRIGGER_COOLDOWN_SECONDS", "60"))
last_trigger_time = 0

SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detect"))
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print("Created directory:", SAVE_DIR)


def clear_folder_except(filename_to_keep):
    """Delete all files in SAVE_DIR except for the specified filename."""
    for fname in os.listdir(SAVE_DIR):
        full_path = os.path.join(SAVE_DIR, fname)
        if full_path != filename_to_keep:
            try:
                os.remove(full_path)
                print("Deleted:", full_path)
            except Exception as e:
                print("Error deleting file:", full_path, e)


def save_full_frame(frame):
    """Save the full frame to SAVE_DIR and return its path."""
    timestamp = int(time.time() * 1000)
    filename = os.path.join(SAVE_DIR, f"full_frame_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print("Saved full frame as", filename)
    return filename


def crop_and_resize_face(input_path, output_path):
    """
    Rotate the image 180 degrees, save the rotated image for inspection,
    convert it to RGB for MTCNN, then crop the face using MTCNN,
    resize the cropped face to 112x112, and save the final image.
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to read image {input_path}")
        return False

    rotated_bgr = cv2.rotate(img, cv2.ROTATE_180)

    timestamp = int(time.time() * 1000)
    rotated_filename = os.path.join(SAVE_DIR, f"rotated_{timestamp}.jpg")
    cv2.imwrite(rotated_filename, rotated_bgr)
    print(f"Saved rotated image as {rotated_filename}")

    rotated_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(rotated_rgb)
    if results:
        x, y, width, height = results[0]['box']
        x, y = max(0, x), max(0, y)
        face_rgb = rotated_rgb[y:y+height, x:x+width]

        face_resized_rgb = cv2.resize(face_rgb, (112, 112))

        face_resized_bgr = cv2.cvtColor(face_resized_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, face_resized_bgr)
        print(f"Cropped, resized, and saved face image as {output_path}")
        return True
    else:
        print(f"Error: No face detected in {input_path}")
        return False


def generate_frames():
    global last_trigger_time
    while True:
        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        current_time = time.time()
        if len(faces) > 0 and (current_time - last_trigger_time) >= COOLDOWN:

            full_frame_path = save_full_frame(frame)

            cropped_face_path = os.path.join(SAVE_DIR, "detected_face_latest.jpg")
            success = crop_and_resize_face(full_frame_path, cropped_face_path)

            os.remove(full_frame_path)

            if success:
                clear_folder_except(cropped_face_path)
                try:
                    response = requests.get(C_TRIGGER_URL, timeout=0.5)
                    print("Trigger sent, response:", response.text)
                    last_trigger_time = current_time  # Update cooldown timer.
                except Exception as e:
                    print("Error sending trigger:", e)
            else:
                print("Face cropping failed; trigger not sent.")

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host=host, port=port, threaded=True)
