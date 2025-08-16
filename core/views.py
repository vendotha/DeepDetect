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
import io

# -------------------------------
# Load Hugging Face pipeline from LOCAL path
# -------------------------------
# Avoid double "deepfake_detector-main" and fix Windows backslashes
model_path = os.path.join(settings.BASE_DIR, "deepfake-detector-faceforensics")
model_path = model_path.replace("\\", "/")

detector_pipe = pipeline(
    task="image-classification",
    model=model_path,
    tokenizer=model_path  # some models need tokenizer too
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def extract_thumbnail(video_path):
    """Extract the first frame of a video as a thumbnail (PIL Image)."""
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

            # Save uploaded file
            filename = default_storage.save(media_file.name, media_file)
            media_path = os.path.join(settings.MEDIA_ROOT, filename)
            media_url = f"{settings.MEDIA_URL}{filename}"  # safe URL
            preview_url = None

            # Analyze media
            result, confidence, explanation = None, None, None
            gemini_input = None

            # -------------------- IMAGE --------------------
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = Image.open(media_path).convert("RGB")
                predictions = detector_pipe(image)
                result = predictions[0]['label']
                confidence = predictions[0]['score']
                gemini_input = image

            # -------------------- VIDEO --------------------
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                thumbnail = extract_thumbnail(media_path)
                if thumbnail:
                    predictions = detector_pipe(thumbnail)
                    result = predictions[0]['label']
                    confidence = predictions[0]['score']

                    thumb_name = filename.rsplit('.', 1)[0] + "_thumb.jpg"
                    thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_name)
                    thumbnail.save(thumb_path)
                    preview_url = f"{settings.MEDIA_URL}{thumb_name}"
                    gemini_input = thumbnail
                else:
                    result = "Unable to extract thumbnail"
                    gemini_input = None

            # -------------------- OTHER FILE --------------------
            else:
                result = "Unsupported file type"
                gemini_input = None

            # -------------------- GEMINI EXPLANATION --------------------
            if gemini_input and result and confidence is not None:
                try:
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    gemini_prompt = (
                        f"The uploaded {'video' if ext in ['.mp4', '.avi', '.mov', '.mkv'] else 'image'} "
                        f"has been checked and it is **{result.upper()}** with a confidence of "
                        f"**{confidence * 100:.2f}%**.\n"
                        f"Now give a simple and confident explanation in bullet points about why this is considered "
                        f"{result.lower()}. Avoid technical terms. No AI mentions. No probabilities. "
                        f"Just state clearly why it is {result.lower()}."
                    )
                    gemini_response = gemini_model.generate_content([gemini_input, gemini_prompt])
                    explanation = gemini_response.text
                except Exception as e:
                    explanation = f"Gemini Error: {e}"

            # Save results to session
            request.session['result'] = result
            request.session['confidence'] = confidence
            request.session['explanation'] = explanation
            request.session['uploaded_file_url'] = media_url
            request.session['preview'] = preview_url

            return redirect('analyze')  # Redirect to result page
    else:
        form = UploadForm()

    return render(request, 'core/home.html', {'form': form})


def analyze(request):
    """Display analysis results."""
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
