FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r cat.txt
EXPOSE 8080
CMD ["gunicorn" , "app:app"]


