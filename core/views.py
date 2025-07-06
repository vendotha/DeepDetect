import os
import mimetypes
from django.shortcuts import render
from django.conf import settings
from .forms import UploadForm
from transformers import pipeline
import google.generativeai as genai
from PIL import Image
import io
import cv2

# Load Hugging Face pipeline
detector_pipe = pipeline("image-classification", model="HrutikAdsare/deepfake-detector-faceforensics")

# Configure Gemini
genai.configure(api_key="AIzaSyDMV42jvwDykHB-Usfua9P1qRT78Fjs__U")


def extract_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        # Convert BGR to RGB and return PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image
    return None


def home(request):
    result = None
    confidence = None
    explanation = None
    uploaded_file_url = None
    preview_url = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            media_file = form.cleaned_data['media_file']
            media_path = os.path.join(settings.MEDIA_ROOT, media_file.name)

            # Save uploaded file
            with open(media_path, 'wb+') as f:
                for chunk in media_file.chunks():
                    f.write(chunk)

            uploaded_file_url = os.path.join(settings.MEDIA_URL, media_file.name)

            ext = os.path.splitext(media_file.name)[1].lower()

            # --- Prediction ---
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = Image.open(media_path).convert("RGB")
                predictions = detector_pipe(image)
                result = predictions[0]['label']
                confidence = predictions[0]['score']
                gemini_input = image

            elif ext == '.mp4':
                thumbnail = extract_thumbnail(media_path)
                if thumbnail:
                    predictions = detector_pipe(thumbnail)
                    result = predictions[0]['label']
                    confidence = predictions[0]['score']
                    # Save thumbnail to MEDIA for preview
                    thumb_filename = media_file.name.replace('.mp4', '_thumb.jpg')
                    thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_filename)
                    thumbnail.save(thumb_path)
                    preview_url = os.path.join(settings.MEDIA_URL, thumb_filename)
                    gemini_input = thumbnail
                else:
                    result = "Could not extract thumbnail from video"
                    gemini_input = None
            else:
                result = "Unsupported file type"
                gemini_input = None

            # --- Gemini Explanation (Simple, bullet points, confident, no AI hints) ---
            if gemini_input and result and confidence is not None:
                try:
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    gemini_prompt = (
                        f"The uploaded {'video' if ext == '.mp4' else 'image'} has been checked and it is "
                        f"**{result.upper()}** with a confidence of **{confidence * 100:.2f}%**.\n"
                        "Now give a simple and confident explanation in bullet points about why this is considered "
                        f"{result.lower()}. Avoid technical terms. The explanation should sound like a human is saying it "
                        "and not an AI. Don't include any uncertain language. Be sure and confident. The points should be easy "
                        "for anyone to understand, even without education. Do not mention anything about the model, AI, or probabilities. "
                        "Just say why it's {result.lower()}."
                    )
                    gemini_response = gemini_model.generate_content([gemini_input, gemini_prompt])
                    explanation = gemini_response.text
                except Exception as e:
                    explanation = f"Gemini Error: {e}"



    else:
        form = UploadForm()

    return render(request, 'core/home.html', {
        'form': form,
        'result': result,
        'confidence': confidence,
        'explanation': explanation,
        'uploaded_file_url': uploaded_file_url,
        'preview': preview_url,
    })
