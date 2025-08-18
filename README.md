# 🕵️ DeepDetect - AI-Powered Deepfake Detection
<!-- Add a screenshot later -->

DeepDetect is an advanced web application that uses machine learning to detect deepfake images and videos with high accuracy. It combines the power of Hugging Face's image classification models with Google's Gemini AI for detailed explanations.

## ✨ Features

- **AI-Powered Detection**: Uses state-of-the-art deep learning models
- **Multi-Format Support**: Works with images (JPG, PNG) and videos (MP4)
- **Detailed Analysis**: Provides confidence scores and human-readable explanations
- **User-Friendly Interface**: Clean, modern UI with theme switching
- **Privacy Focused**: Files are processed temporarily and not stored permanently

## 🛠️ Technologies Used

- **Backend**: Django 5.2
- **Frontend**: HTML5, CSS3, JavaScript
- **AI Models**:
  - Hugging Face `deepfake-detector-faceforensics`
  - Google Gemini 1.5 Flash
- **Computer Vision**: OpenCV for video processing

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip
- Google Gemini API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
Create and activate virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Set up environment variables:
bash
cp .env.example .env
Edit .env with your API keys:

text
GEMINI_API_KEY=your_api_key_here
Run migrations:
bash
python manage.py migrate
Start development server:
bash
python manage.py runserver
📂 Project Structure

text
deepfake-detector/
├── core/                  # Main app
│   ├── templates/         # HTML templates
│   ├── views.py           # Business logic
│   ├── urls.py            # App URLs
│   └── ...
├── deepfake_detector/     # Project config
├── media/                 # Uploaded files
├── manage.py
└── requirements.txt
🌐 Usage

Access the web interface at http://localhost:8000
Upload an image or video file
View the analysis results including:
Real/Fake prediction
Confidence percentage
Detailed AI explanation
Media preview
🤖 AI Models

The system uses two AI models:

Detection Model:
HrutikAdsare/deepfake-detector-faceforensics from Hugging Face
Specialized in detecting facial manipulations
Explanation Model:
Google Gemini 1.5 Flash
Generates human-readable explanations of the detection results
📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

Hugging Face for the deepfake detection model
Google for the Gemini API
Django community for the web framework
