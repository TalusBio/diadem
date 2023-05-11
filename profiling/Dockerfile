# FROM python:3.9-slim
FROM --platform=linux/amd64 python:3.9-bullseye

RUN apt-get update && apt-get install -y build-essential gcc python3-dev
RUN python3 -m pip install git+https://github.com/jspaezp/alphatims.git@feature/dockerfile

ENTRYPOINT [ "alphatims" ]
