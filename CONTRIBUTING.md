# Contributing to OAAX OpenVINO implementation

Thank you for your interest in contributing to OAAX's OpenVINO implementation! We welcome contributions from the community to improve the current implementation of the OAAX runtime and conversion toolchain.

Before you start contributing, please take a moment to read through [OAAX general CONTRIBUTING document](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md). It sets out the guidelines for contributing to OAAX projects, including code style, commit message format, and other important information.

## Table of contents
- [How to contribute](#how-to-contribute)
- [Development environment](#development-environment)

## How to contribute

The OAAX OpenVINO implementation is built to be able to make use of Intel's CPU, GPU, and NPU for model inference. If you'd like to suggest improvements to the documentation or implementation, report a bug, or contribute a new feature, please follow these steps:
1. **Open an issue** in the repository to discuss your idea or improvement.
2. **Fork the repository** and create a new branch for your changes.
3. **Make your changes** and commit them with a clear message.
4. **Push your changes** to your forked repository.
5. **Submit a pull request** to the main repository for review.


## Development environment

The OAAX OpenVINO implementation has been tested for building on x86_64 architecture machines running Ubuntu 22.04 or later.

### Runtime

The runtime library is built using CMake and several other dependencies. You can easily install them on a host machine by running this command (assuming you have cloned the repository):

```bash
sudo bash scripts/setup-env.sh
```

### Conversion toolchain

The conversion toolchain is built using Docker and requires Docker to be installed on your machine. You can find instructions for installing Docker [here](https://docs.docker.com/get-docker/).