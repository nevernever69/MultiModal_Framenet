#!/bin/bash

# Exit the script if any command fails
set -e

# Inform the user that the download is starting
echo "Downloading dataset..."

# Download the zip file using curl with a progress bar
curl --progress-bar "https://drive.usercontent.google.com/download?id=1isX-1QrlhS8AcH3KTGjwtdmXN-6Ab0Ba&confirm=t&uuid=df5788f4-e65d-481f-ba2e-e9e0bfd0eeba" -o images.zip

# Inform the user that extraction is starting
echo "Extracting dataset into images folder..."

# Create the images folder if it doesn’t already exist
if [ ! -d "images" ]; then
    mkdir images
fi

# Extract the contents of the zip file into the images folder, avoiding overwriting existing files
unzip -n images.zip -d images/



count=$(ls -1 images/ | wc -l)
if [ "$count" -eq 1 ]; then
    item=$(ls -1 images/)
    if [ -d "images/$item" ]; then
        mv "images/$item"/* images/
        rmdir "images/$item"
    fi
fi

# Confirm completion
echo "Done."
