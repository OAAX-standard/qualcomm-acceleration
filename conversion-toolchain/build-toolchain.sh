set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2 &>/dev/null || true
mkdir build

# Build the toolchain as a Docker image
docker build -t qnn-toolchain:latest .

# Save the Docker image as a tarball
docker save qnn-toolchain:latest -o ./build/qnn-toolchain-latest.tar

# You can run the conversion toolchain using the following command:
# `./input` should contain the `input.zip` file
# `./output` will contain the output files if the conversion is successful
docker run --rm \
    -v ./input:/app/input \
    -v ./output:/app/output \
    qnn-toolchain:latest \
    /app/input/input.zip \
    /app/output
