from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import default_storage
from .forms import UploadForm
from .gemini_config import GEMINI_API_KEY
from transformers import pipeline
import google.generativeai as genai
from PIL import Image
import os
import cv2

# Load HF pipeline
detector_pipe = pipeline("image-classification", model="HrutikAdsare/deepfake-detector-faceforensics")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def extract_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

def home(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            media_file = form.cleaned_data['media_file']
            ext = os.path.splitext(media_file.name)[1].lower()
            filename = default_storage.save(media_file.name, media_file)
            media_path = os.path.join(settings.MEDIA_ROOT, filename)
            media_url = os.path.join(settings.MEDIA_URL, filename)
            preview_url = None

            # Analyze media
            result, confidence, explanation = None, None, None
            gemini_input = None

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
                    thumb_name = filename.replace('.mp4', '_thumb.jpg')
                    thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_name)
                    thumbnail.save(thumb_path)
                    preview_url = os.path.join(settings.MEDIA_URL, thumb_name)
                    gemini_input = thumbnail
                else:
                    result = "Unable to extract thumbnail"
                    gemini_input = None
            else:
                result = "Unsupported file type"
                gemini_input = None

            # Generate Gemini explanation
            if gemini_input and result and confidence is not None:
                try:
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    gemini_prompt = (
                        f"The uploaded {'video' if ext == '.mp4' else 'image'} has been checked and it is "
                        f"**{result.upper()}** with a confidence of **{confidence * 100:.2f}%**.\n"
                        f"Now give a simple and confident explanation in bullet points about why this is considered "
                        f"{result.lower()}. Avoid technical terms. No AI mentions. No probabilities. Just state clearly why it is {result.lower()}."
                    )
                    gemini_response = gemini_model.generate_content([gemini_input, gemini_prompt])
                    explanation = gemini_response.text
                except Exception as e:
                    explanation = f"Gemini Error: {e}"

            # Save to session
            request.session['result'] = result
            request.session['confidence'] = confidence
            request.session['explanation'] = explanation
            request.session['uploaded_file_url'] = media_url
            request.session['preview'] = preview_url

            return redirect('analyze')  # Point to the analyze view
    else:
        form = UploadForm()

    return render(request, 'core/home.html', {'form': form})


def analyze(request):
    result = request.session.get('result')
    confidence = request.session.get('confidence')
    explanation = request.session.get('explanation')
    uploaded_file_url = request.session.get('uploaded_file_url')
    preview = request.session.get('preview')

    context = {
        'result': result,
        'confidence': confidence,
        'explanation': explanation,
        'uploaded_file_url': uploaded_file_url,
        'preview': preview,
    }

    return render(request, 'core/result.html', context)
