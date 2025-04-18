#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Ensure s3cmd is installed
if ! command -v s3cmd &>/dev/null; then
    echo "Error: s3cmd is not installed. Please install it and try again."
    exit 1
fi

# Ensure at least two arguments are provided: remote path and at least one file
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <remote-path> <file1> [<file2> ... <fileN>]"
    exit 1
fi

# Extract the remote-path argument
remote_path=$1
shift # Shift arguments to access the file list

OBJECT_KEY="s3://oaax/$remote_path/"

# Check if s3cmd is configured properly
if ! s3cmd --help &>/dev/null; then
    echo "Error: s3cmd is not configured. Please configure it and try again."
    exit 1
fi

# Upload files to the object storage using s3cmd
echo "Uploading files to $OBJECT_KEY..."
s3cmd put "$@" "$OBJECT_KEY"

echo "Upload completed successfully."
