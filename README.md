# qualcomm-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Qualcomm SOC on arm64 machines.

## Artifacts

- The OAAX runtime is available as a shared library that can be used by developers to load and run optimized models on an Qualcomm SOC.

## Usage

### Using the runtime library

To use the runtime library, you need to have the Hailo driver installed on the X86_64 machine.

The OAAX QNN runtime can be used just like the other OAAX runtimes. You can find various and diverse usage examples in
the [examples](https://github.com/oaax-standard/examples) repository.