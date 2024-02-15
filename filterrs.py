import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter
import pilgram,pilgram.css

def apply_filter_to_whole_image(image, filter_type):
    if filter_type == 'brightness':
        image = pilgram.css.brightness(image)  # Convert to grayscale
    elif filter_type == 'grayscale':
        image = pilgram.css.grayscale(image)  # Apply blur filter
    elif filter_type == 'saturate':
        image = pilgram.css.saturate(image)  # Apply blur filter
    elif filter_type == 'sepia':
        image = pilgram.css.sepia(image)  # Apply blur filter
    elif filter_type == '_1977':
        image = pilgram._1977(image)  # Apply blur filter
    elif filter_type == 'aden':
        image = pilgram.aden(image)  # Apply blur filter
    elif filter_type == 'brannan':
        image = pilgram.brannan(image)  # Apply blur filter
    elif filter_type == 'brooklyn':
        image = pilgram.brooklyn(image)  # Apply blur filter
    elif filter_type == 'clarendon':
        image = pilgram.clarendon(image)  # Apply blur filter
    elif filter_type == 'earlybird':
        image = pilgram.earlybird(image)  # Apply blur filter
    elif filter_type == 'gingham':
        image = pilgram.gingham(image)  # Apply blur filter
    elif filter_type == 'hudson':
        image = pilgram.hudson(image)  # Apply blur filter
    elif filter_type == 'inkwell':
        image = pilgram.inkwell(image)  # Apply blur filter
    elif filter_type == 'kelvin':
        image = pilgram.kelvin(image)  
    elif filter_type == 'lark':
        image = pilgram.lark(image)  
    elif filter_type == 'lofi':
        image = pilgram.lofi(image)  
    elif filter_type == 'maven':
        image = pilgram.maven(image)  
    elif filter_type == 'mayfair':
        image = pilgram.mayfair(image)  
    elif filter_type == 'moon':
        image = pilgram.moon(image)  
    elif filter_type == 'nashville':
        image = pilgram.nashville(image)  
    elif filter_type == 'perpetua':
        image = pilgram.perpetua(image)  
    elif filter_type == 'reyes':
        image = pilgram.reyes(image)  
    elif filter_type == 'rise':
        image = pilgram.rise(image)  
    elif filter_type == 'slumber':
        image = pilgram.slumber(image)  
    elif filter_type == 'stinson':
        image = pilgram.stinson(image)  
    elif filter_type == 'toaster':
        image = pilgram.toaster(image)  
    elif filter_type == 'valencia':
        image = pilgram.valencia(image)  
    elif filter_type == 'walden':
        image = pilgram.walden(image)  
    elif filter_type == 'willow':
        image = pilgram.willow(image)  
    elif filter_type == 'xpro2':
        image = pilgram.xpro2(image)  
    return image
     



def apply_filter_to_face(cv_image, filter_type):
    # Detect face locations in the image
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

# Detect face landmarks
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)

# Extract face landmarks
    if results.multi_face_landmarks:
     for face_landmarks in results.multi_face_landmarks:
        # Convert normalized landmark coordinates to pixel coordinates
        image_height, image_width, _ = cv_image.shape
        landmark_points = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            landmark_points.append((x, y))
# Apply the convex hull algorithm to get the convex hull points
    hull = cv2.convexHull(np.array(landmark_points), returnPoints=True)

# Create a mask using the convex hull points
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    sample = cv_image[::]
    cv2.drawContours(mask, [hull],-500, 255,-500)
    cv2.fillConvexPoly(sample, hull, 0)

# cv2.imshow("sample",sample)

# Apply the filter (e.g., blur) using the mask
    filtered_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    filtered_pil_image = Image.fromarray(filtered_image)
    if filter_type == 'blur':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.BLUR)
    elif filter_type == 'smooth':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.SMOOTH)  
    elif filter_type == 'smooth_more':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.SMOOTH_MORE)
    elif filter_type == 'edge_enhance':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.EDGE_ENHANCE)  
    elif filter_type == 'edge_enhance_more':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)  
    elif filter_type == 'emboss':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.EMBOSS)  
    elif filter_type == 'find_edges':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.FIND_EDGES)  
    elif filter_type == 'sharpen':
                filtered_pil_image = filtered_pil_image.filter(ImageFilter.SHARPEN)  
    
# Convert the filtered image back to NumPy array
    filtered_image_rgb = np.array(filtered_pil_image)
# Convert the filtered image back to BGR for display
    filtered_image_bgr = cv2.cvtColor(filtered_image_rgb, cv2.COLOR_RGB2BGR)
    image = cv2.bitwise_or(sample,filtered_image_bgr)
    return image