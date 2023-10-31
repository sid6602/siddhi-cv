from flask import Flask, render_template, request
import cv2
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = cv2.imread('static/input.jpg')

        transformation_type = request.form['algorithm']

        if transformation_type == 'Thresholding':
            # Define and apply the Thresholding algorithm
            _, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            transformed_image = thresholded_image
        elif transformation_type == 'Contour':
            # Define and apply the Contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            transformed_image = np.copy(image)
            cv2.drawContours(transformed_image, contours, -1, (0, 255, 0), 2)

        elif transformation_type == 'Watershed':
            # Define and apply the watershed algorithm
            transformed_image = np.copy(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0 but 1
            markers = markers + 1

            # Mark the region of unknown with 0
            markers[unknown == 255] = 0

            # Apply watershed
            markers = cv2.watershed(image, markers)
            transformed_image[markers == -1] = [255, 0, 0]  # Color boundaries in blue

        elif transformation_type == 'Grabcut':
            # Define and apply the grabcut algorithm
            transformed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a mask
            mask = np.zeros(transformed_image.shape[:2], np.uint8)

            # Define the background and foreground models
            background_model = np.zeros((1, 65), np.float64)
            foreground_model = np.zeros((1, 65), np.float64)

            # Define the region of interest (ROI) as a rectangle
            rect = (50, 50, 450, 290)

            # Apply GrabCut algorithm
            cv2.grabCut(transformed_image, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create a mask where the background is 0 and the foreground is 1 or 3
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Multiply the original image with the mask to get the segmented image
            transformed_image = transformed_image * mask2[:, :, np.newaxis]
        else:
            # Handle an unknown transformation type
            return "Unknown Algorithm type"

        # Convert the transformed image to base64 for displaying in HTML
        _, buffer = cv2.imencode('.jpg', transformed_image)
        transformed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return render_template('index.html', transformed_image=transformed_image_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
