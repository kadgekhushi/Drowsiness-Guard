"""Optional CNN eye-state classifier (TFLite). Output = probability that the eye is CLOSED."""
import numpy as np


def _load_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=path)


class EyeCNN:
    def __init__(self, path):
        self.it = _load_interpreter(path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def prob_closed(self, crop_gray_64) -> float:
        x = crop_gray_64.astype(np.float32).reshape(1, 64, 64, 1)   # raw 0..255, model rescales itself
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        return float(self.it.get_tensor(self.out["index"]).ravel()[0])
