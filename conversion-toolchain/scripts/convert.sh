set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <onnx-path> <output-directory>" >&2
  exit 1
fi

#DIR="$(cd "$(dirname "$0")" && pwd)"
#cd "$DIR" || exit 1

onnx_path=$1
output_directory=$2

conversion_toolchain -z "$onnx_path" -o "$output_directory"
