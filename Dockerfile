FROM docker.io/library/python:3.11-slim@sha256:3c6d7bbe446b236c2bb03d60c03d4962fb22040c6e14fd23c61447231b362ec7

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*
