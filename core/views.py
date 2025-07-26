import os
import mimetypes
from django.shortcuts import render, redirect
from .gemini_config import GEMINI_API_KEY
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
genai.configure(api_key=GEMINI_API_KEY)


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
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            media_file = form.cleaned_data['media_file']
            media_path = os.path.join(settings.MEDIA_ROOT, media_file.name)

            # Save uploaded file
            with open(media_path, 'wb+') as f:
                for chunk in media_file.chunks():
                    f.write(chunk)

            # Store filename in session and redirect
            request.session['analyze_filename'] = media_file.name
            return redirect('analyze')
    else:
        form = UploadForm()

    return render(request, 'core/home.html', {'form': form})

def analyze(request):
    # Initialize all variables with default values
    result = "unknown"
    confidence = 0.0
    explanation = "Analysis not available"
    uploaded_file_url = None
    preview_url = None  # Initialize with None
    gemini_input = None

    # Get filename from session
    filename = request.session.get('analyze_filename')
    if not filename:
        return redirect('home')

    media_path = os.path.join(settings.MEDIA_ROOT, filename)
    uploaded_file_url = os.path.join(settings.MEDIA_URL, filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
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
                thumb_filename = filename.replace('.mp4', '_thumb.jpg')
                thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_filename)
                thumbnail.save(thumb_path)
                preview_url = os.path.join(settings.MEDIA_URL, thumb_filename)
                gemini_input = thumbnail
            else:
                result = "video_processing_error"
                explanation = "Could not extract thumbnail from video"

        else:
            result = "unsupported_file_type"
            explanation = f"Unsupported file type: {ext}"

        # --- Gemini Explanation ---
        if gemini_input and result not in ["video_processing_error", "unsupported_file_type"]:
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
                explanation = f"Explanation generation error: {str(e)}"

    except Exception as e:
        result = "processing_error"
        explanation = f"An error occurred during processing: {str(e)}"

    return render(request, 'core/result.html', {
        'result': result,
        'confidence': confidence,
        'explanation': explanation,
        'uploaded_file_url': uploaded_file_url,
        'preview': preview_url,  # Now always defined
    })