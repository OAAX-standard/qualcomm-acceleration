ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.10.16 as base

WORKDIR /app

COPY onnxruntime_qnn-1.19.0-cp310-cp310-linux_x86_64.whl /app/onnxruntime_qnn-1.19.0-cp310-cp310-linux_x86_64.whl
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
COPY conversion_toolchain /app/conversion_toolchain
COPY scripts/ /app/scripts

RUN pip install . ./onnxruntime_qnn-1.19.0-cp310-cp310-linux_x86_64.whl --no-cache-dir

ENTRYPOINT ["bash", "/app/scripts/convert.sh"]