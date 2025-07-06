from django.shortcuts import render
from .forms import UploadForm
from .utils.model import predict_image, predict_video
import os
from django.conf import settings

def home(request):
    result = None
    confidence = None

    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['media_file']
            filepath = os.path.join(settings.MEDIA_ROOT, file.name)

            with open(filepath, 'wb+') as dest:
                for chunk in file.chunks():
                    dest.write(chunk)

            if file.content_type.startswith('image'):
                result, confidence = predict_image(filepath)
            elif file.content_type.startswith('video'):
                result, confidence = predict_video(filepath)
            else:
                result = "Unsupported file type."

            os.remove(filepath)
    else:
        form = UploadForm()

    return render(request, 'core/home.html', {'form': form, 'result': result, 'confidence': confidence})
