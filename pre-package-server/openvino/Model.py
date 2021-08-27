import glob
import os
from io import BytesIO

import cv2
import numpy as np
from openvino.inference_engine import IECore


class Model:
    def __init__(self, model_uri):
        ir_xml, ir_bin = '', ''
        if glob.glob(os.path.join(model_uri, '*.xml')):
            ir_xml = glob.glob(os.path.join(model_uri, '*.xml'))[0]
        if glob.glob(os.path.join(model_uri, '*.bin')):
            ir_bin = glob.glob(os.path.join(model_uri, '*.bin'))[0]

        ie = IECore()
        net = ie.read_network(ir_xml, ir_bin)
        self.model = ie.load_network(net, "CPU")
        self.input_key = list(self.model.input_info)[0]
        self.output_key = list(self.model.outputs.keys())[0]

    def predict(self, X, feature_names=None, meta=None):
        if isinstance(X, bytes):
            image = cv2.imdecode(np.frombuffer(BytesIO(X).getbuffer(), np.uint8), -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(image, (224, 224)) / 255
            input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
            result = self.model.infer(inputs={self.input_key: input_image})[
                self.output_key
            ]
            return result
