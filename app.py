
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering, logging as tf_logging
from PIL import Image
import torch
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import base64
from flask import Flask, request, render_template_string, send_file
import os

port = int(os.environ.get("PORT", 7860))

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    local_files_only=False
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    local_files_only=False
)

print("Loading BLIP VQA model...")
vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base",
    local_files_only=False
)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    local_files_only=False
)
print("All models loaded successfully!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>lab - 7</title>

</head>
<body>
    <h1>Lab 7: Multimodal Pipeline </h1>
    <p>upload Image for nepali caption and then and listen on nepali</p>

    <div class="upload-form">
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Process Image</button>
        </form>
    </div>

    {% if result %}
    <div class="result">
        <img src="data:image/jpeg;base64,{{ image_data }}" alt="Uploaded Image">

        <h2> Caption</h2>
        <p class="caption english"><strong>English</strong> {{ result.caption }}</p>
        <p class="caption nepali"><strong>Nepali</strong> {{ result.translated }}</p>

        <h2> Audio (Nepali)</h2>
        <audio controls>
            <source src="data:audio/mp3;base64,{{ result.audio_data }}" type="audio/mp3">
             support the audio 
        </audio>


    </div>
    {% endif %}

    <hr>

    <h1>Vision: Visual Question Answering (VQA)</h1>
    <p>Upload an image and ask a question about it</p>

    <div class="upload-form">
        <form method="POST" action="/vqa" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*">
            <button type="submit">Process Image</button>
        </form>

        {% if vqa_image_data %}
        <div style="margin-top:10px; margin-bottom:8px;">
            <img src="data:image/jpeg;base64,{{ vqa_image_data }}" alt="Processed Image" style="max-width:200px; display:block;">
        </div>
        <form method="POST" action="/vqa">
            <input type="hidden" name="image_data" value="{{ vqa_image_data }}">
            <br>
            <input type="text" name="question" placeholder="Ask a question about the image..." required style="width: 300px; padding: 5px;">
            <button type="submit">Ask</button>
        </form>
        {% endif %}
    </div>

    {% if vqa_result %}
    <div class="result">
        <img src="data:image/jpeg;base64,{{ vqa_image_data }}" alt="Uploaded Image" style="max-width: 400px;">

        <h2>Question</h2>
        <p><strong>{{ vqa_result.question }}</strong></p>

        <h2>Answer</h2>
        <p>{{ vqa_result.answer }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_data = None
    vqa_result = None
    vqa_image_data = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            translated = GoogleTranslator(source='en', target='ne').translate(caption)

            tts = gTTS(text=translated, lang="ne")
            mp3_bytes = BytesIO()
            tts.write_to_fp(mp3_bytes)
            mp3_bytes.seek(0)
            audio_data = base64.b64encode(mp3_bytes.read()).decode()

            result = {
                "caption": caption,
                "translated": translated,
                "audio_data": audio_data
            }

    return render_template_string(HTML_TEMPLATE, result=result, image_data=image_data,
                                  vqa_result=vqa_result, vqa_image_data=vqa_image_data)


@app.route("/vqa", methods=["POST"])
def vqa():
    vqa_result = None
    vqa_image_data = None

    file = request.files.get("image")
    image_data_b64 = request.form.get("image_data")
    question = request.form.get("question", "").strip()

    if file and not question:
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        vqa_image_data = base64.b64encode(buffered.getvalue()).decode()


    elif question and (file or image_data_b64):
        if file:
            image_bytes = file.read()
        else:
            image_bytes = base64.b64decode(image_data_b64)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        vqa_image_data = base64.b64encode(buffered.getvalue()).decode()

        inputs = vqa_processor(image, question, return_tensors="pt")
        with torch.no_grad():
            out = vqa_model.generate(**inputs)
        answer = vqa_processor.decode(out[0], skip_special_tokens=True)

        vqa_result = {
            "question": question,
            "answer": answer
        }

    return render_template_string(HTML_TEMPLATE, result=None, image_data=None,
                                  vqa_result=vqa_result, vqa_image_data=vqa_image_data)


if __name__ == "__main__":
    print("Starting Multimodal Pipeline Web App...")
    print(" http://localhost:7860 to run locally")


    app.run(host="0.0.0.0", port=port)