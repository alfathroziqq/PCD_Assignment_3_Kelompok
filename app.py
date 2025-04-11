from flask import Flask, render_template, request
import os
import cv2
from sobel import sobel
from prewitt import prewitt
from robert import roberts_edge_detection
from log_filter import LoGFilter
from compass import compass_operator

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(name, img_array):
    path = os.path.join(UPLOAD_FOLDER, name)
    cv2.imwrite(path, img_array)
    return path

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        img_file = request.files["image"]
        method = request.form["method"]

        img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
        img_file.save(img_path)

        if method == "sobel":
            original, sobel_x, sobel_y, magnitude = sobel(img_path)
            results.append(("Original", '/' + save_image("sobel_orig.png", original)))
            results.append(("Sobel X", '/' + save_image("sobel_x.png", sobel_x)))
            results.append(("Sobel Y", '/' + save_image("sobel_y.png", sobel_y)))
            results.append(("Magnitude", '/' + save_image("sobel_mag.png", magnitude)))

        elif method == "prewitt":
            original, prewitt_x, prewitt_y, combined = prewitt(img_path)
            results.append(("Original", '/' + save_image("prewitt_orig.png", original)))
            results.append(("Prewitt X", '/' + save_image("prewitt_x.png", prewitt_x)))
            results.append(("Prewitt Y", '/' + save_image("prewitt_y.png", prewitt_y)))
            results.append(("Combined", '/' + save_image("prewitt_combined.png", combined)))
        
        elif method == "roberts":
            original, result = roberts_edge_detection(img_path)
            results.append(("Original", '/' + save_image("robert_orig.png", original)))
            results.append(("Roberts Edge", '/' + save_image("robert_edge.png", result)))

        elif method == "log":
            original, filtered = LoGFilter(img_path)
            results.append(("Original", '/' + save_image("log_orig.png", original)))
            results.append(("LoG Filter", '/' + save_image("log_filtered.png", filtered)))

        elif method == "compass":
            from PIL import Image
            import numpy as np

            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            directions = [
                'Utara', 'Timur Laut', 'Timur', 'Tenggara',
                'Selatan', 'Barat Daya', 'Barat', 'Barat Laut'
            ]

            for dir_name in directions:
                edge_img = compass_operator(img, dir_name)
                edge_arr = np.array(edge_img)
                fname = f"compass_{dir_name.replace(' ', '_').lower()}.png"
                results.append((dir_name, '/' + save_image(fname, edge_arr)))

            combined = np.array(compass_operator(img, direction='all'))
            results.append(("Gabungan", '/' + save_image("compass_combined.png", combined)))

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)