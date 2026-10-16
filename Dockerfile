# Equivalent of the submitted image. The submission itself is assembled with build/assemble.sh,
# which appends the same single layer to the unmodified base layers without a local docker pull.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

COPY app/requirements.txt /app/requirements.txt
# --no-deps: nothing here may replace the ROCm torch of the base image
RUN pip install --no-cache-dir --no-deps --target /opt/pinpoint/deps -r /app/requirements.txt
COPY app/ /app/
RUN mkdir -p /app/output /models

ENV PYTHONPATH=/opt/pinpoint/deps \
    PINPOINT_MODEL_DIR=/models \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
CMD ["python3", "/app/app.py", "--serve"]
