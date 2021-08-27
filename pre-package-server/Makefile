VERSION := v0.3.0-dev
IMAGE_NAME_BASE=openvino-prepackaged
IMAGE_BASE=evenwei/${IMAGE_NAME_BASE}

build:
	docker build . -t ${IMAGE_BASE}:${VERSION}

push:
	docker push ${IMAGE_BASE}:${VERSION}
