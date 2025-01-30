import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from pathlib import Path
from lgg import logger
import tempfile
import uuid

from .data_reader import DataReader
from ..config_utils import ModelConfig
from ..utils import get_random_temp_path
    
def quantize_onnx(input_onnx: Path, data_reader: DataReader, config: ModelConfig) -> Path:
    logger.info(f"Quantizing ONNX model")
    
    # Generate a random path for the output ONNX file
    output_onnx = get_random_temp_path()

    # Pre-process the original float32 model.
    model_changed = qnn_preprocess_model(input_onnx, output_onnx)
    model_to_quantize = output_onnx if model_changed else input_onnx
    
    if model_changed:
        logger.debug(f"Model was pre-processed")
    else:
        logger.debug(f"Model pre-processing failed")
    
    # Generate a suitable quantization configuration for this model.
    # Note that we're choosing to use uint16 activations and uint8 weights.
    activation_type = QuantType.QUInt16
    weight_type = QuantType.QUInt8
    qnn_config = get_qnn_qdq_config(model_to_quantize,
                                    data_reader,
                                    activation_type=activation_type,    # uint16 activations
                                    weight_type=weight_type)            # uint8 weights
    
    # Quantize the model.
    quantize(model_to_quantize, output_onnx, qnn_config)
    
    return output_onnx

