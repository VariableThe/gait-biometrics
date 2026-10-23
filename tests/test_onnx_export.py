"""Real graph export/checking without undeclared onnxscript dependencies."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from models.cnn_lstm import DeepConvLSTM
from models.compress import export_onnx


class OnnxExportTests(unittest.TestCase):
    def test_deepconvlstm_exports_a_checker_valid_dynamic_batch_graph(self):
        torch.set_num_threads(1)
        model = DeepConvLSTM(3, 2, hidden_dim=8, n_lstm_layers=1, dropout=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            export_onnx(model, str(path), (1, 3, 32), opset=18)
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for batch_size in (1, 2):
                inputs = torch.randn(batch_size, 3, 32)
                with torch.inference_mode():
                    expected = model(inputs).numpy()
                actual = session.run(None, {"input": inputs.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(
            graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch"
        )


if __name__ == "__main__":
    unittest.main()
