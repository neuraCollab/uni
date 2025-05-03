#! /bin/bash

# install_script.sh

# First param - path_to_script (required)
# You can pass n args in instalation_script 
install_script() {
    path_to_script="$1"
    filename=$(basename "$path")

    echo "Starting to install $filename ..."
    . $path_to_script "${@:2}"
}