from flask import Flask, render_template, request
import os
import cv2
from prewitt import prewitt
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

        if method == "prewitt":
            original, prewitt_x, combined = prewitt(img_path)
            results.append(("Original", '/' + save_image("prewitt_orig.png", original)))
            results.append(("Prewitt X", '/' + save_image("prewitt_x.png", prewitt_x)))
            results.append(("Combined", '/' + save_image("prewitt_combined.png", combined)))

        elif method == "log":
            original, filtered = LoGFilter(img_path)
            results.append(("Original", '/' + save_image("log_orig.png", original)))
            results.append(("LoG Filter", '/' + save_image("log_filtered.png", filtered)))

        elif method == "compass":
            original, edge_maps, combined = compass_operator(img_path)
            results.append(("Original", '/' + save_image("compass_orig.png", original)))
            for k, v in edge_maps.items():
                results.append((k, '/' + save_image(f"compass_{k}.png", v)))
            results.append(("Gabungan", '/' + save_image("compass_combined.png", combined)))

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
