import os
import io
import base64
from flask import Flask, render_template, request
from PIL import Image
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDwHlf_LEBVew4POMa9lZYkvhvfkxb73Jk")

app = Flask(__name__)


def classify_image(image):
    """Sends the uploaded image to Gemini 1.5 Flash for plant disease classification."""
    try:
        # Convert image to Base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")  # Ensure correct format
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()

        # Prepare Image Data for Gemini API
        img_data = {
            "mime_type": "image/png",
            "data": img_base64,
        }

        # Call Gemini API for classification
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([img_data, "Identify the plant disease in this image."])

        return response.text if response else "No prediction available", img_base64
    except Exception as e:
        return f"Error: {str(e)}", None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_data = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file)
            prediction, image_data = classify_image(image)
    return render_template("index.html", prediction=prediction, image_data=image_data)


if __name__ == "__main__":
    app.run(debug=True)
