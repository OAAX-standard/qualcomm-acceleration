# OAAX Conversion toolchain

This reference implementation of an OAAX conversion toolchain is a simple example of how a conversion toolchain should be
implemented.

For the sake of example, we'll build a Docker image (toolchain) that converts a non-optimized ONNX model to optimized
ONNX model using the [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) tool.   
The optimization step includes checking the correctness of the ONNX graph, fusing operators, removing unused nodes, and
applying other optimizations. Hence, the resulting model is expected to be smaller and efficient for deployment.

## Getting started

The Docker image is built using the provided [Dockerfile](Dockerfile) and [entrypoint](scripts%2Fconvert.sh) script. 
The entrypoint script takes two parameters:
- The path to the platform-agnostic model.
- The path to the output directory where the optimized model will be saved, along with a logs file that contains
  the conversion process logs.

The conversion toolchain relies on the Python package `conversion_toolchain` which is installed in the Docker image
during the build stage.

To build the example Docker image, run the following command (Note that you need to have Docker installed):
```bash
bash build-toolchain.sh
```

This will build the Docker image, and save it in the `build/` directory.

## Running the conversion toolchain

To run the conversion toolchain, you need to have the Docker image built and an ONNX model file to convert.
To convert a model, run the following command:
```bash
docker run -v /path/to/model-directory:/model  conversion-toolchain:latest /model/model.onnx /model/output
```
That will create a Docker container, mount the model directory to the container, and run the conversion process on the 
model file `model.onnx` (located in `/path/to/model-directory` on the host machine). 
The optimized model will be saved in the `/path/to/model-directory/output/` directory (on the host).