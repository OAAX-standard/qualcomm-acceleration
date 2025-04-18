set -e

cd "$(dirname "$0")" || exit 1

# Read version from the version file
VERSION_FILE="../VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo "Version file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(<"$VERSION_FILE")

rm -rf artifacts 2 &>/dev/null || true
mkdir artifacts

# Build the toolchain as a Docker image
docker build -t oaax-qualcomm-toolchain:$VERSION .

# Save the Docker image as a tarball
docker save oaax-qualcomm-toolchain:$VERSION -o ./artifacts/oaax-qualcomm-toolchain.tar
