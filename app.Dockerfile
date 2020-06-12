FROM python:3.7

RUN echo "Building GQCNN App"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8  

COPY . .
WORKDIR .

RUN ls -la

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y python3-rtree

RUN pip3 install .

RUN pip3 install -r requirements/web_requirements.txt

EXPOSE 5000

ENTRYPOINT python3 main.py