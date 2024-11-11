set -e

cd "$(dirname "$0")" || exit 1

rm -rf build 2&> /dev/null || true
mkdir build

cd build

cmake ..
make -j