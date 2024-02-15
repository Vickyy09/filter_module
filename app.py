import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image, ImageFilter
import cv2
import mediapipe as mp
import numpy as np
from filterrs import apply_filter_to_whole_image ,apply_filter_to_face

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mp_face_detection = mp.solutions.face_detection

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Get filter choices from the form
        filter_type_whole_image = request.form.get('filter_type_whole_image')
        filter_type_face = request.form.get('filter_type_face')

        # Load the image using OpenCV for face detection
        cv_image = cv2.imread(image_path)

        # Apply filters to the whole image and the face
        image = Image.open(image_path)
        image_whole_image_filtered = apply_filter_to_whole_image(image.copy(), filter_type_whole_image)
        image_face_filtered = apply_filter_to_face(cv_image.copy(), filter_type_face)

        # Save the filtered images
        filtered_image_path_whole_image = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_whole_image_' + os.path.basename(image_path))
        filtered_image_path_face = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_face_' + os.path.basename(image_path))
        image_whole_image_filtered.save(filtered_image_path_whole_image)
        cv2.imwrite(filtered_image_path_face, image_face_filtered)

        return render_template('result.html', original_image=file.filename,
                               filtered_image_whole_image=os.path.basename(filtered_image_path_whole_image),
                               filtered_image_face=os.path.basename(filtered_image_path_face))
    else:
        return render_template('error.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
