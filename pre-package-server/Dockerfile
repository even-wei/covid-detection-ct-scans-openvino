FROM python:3.7-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./openvino /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000
EXPOSE 9000

# Define environment variable
ENV MODEL_NAME Model
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --persistence $PERSISTENCE --access-log

