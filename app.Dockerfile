FROM python:3.7

RUN echo "Building GQCNN App"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8  

COPY . /app
WORKDIR /app

RUN ls -la

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y python3-rtree python3-opengl

RUN pip3 install -r requirements/web_requirements.txt

RUN pip3 install .

EXPOSE 5000

ENTRYPOINT python3 main.py