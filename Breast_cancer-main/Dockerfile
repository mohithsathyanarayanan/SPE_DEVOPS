FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r cat.txt
EXPOSE 8000
CMD ["gunicorn" , "app:app"]


