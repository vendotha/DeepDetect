#!/usr/bin/env bash
# build.sh
set -o errexit  # stop if error occurs

pip install -r requirements.txt
python manage.py collectstatic --noinput
python manage.py migrate
