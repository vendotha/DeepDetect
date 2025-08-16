from django import forms

class UploadForm(forms.Form):
    media_file = forms.FileField()
