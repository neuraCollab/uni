#!/bin/bash

current_program_name="anaconda"

sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

cd ~/Downloads/Anaconda3-2023.09-0-Linux-x86_64.sh

curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

bash ~/Downloads/Anaconda3-2023.09-0-Linux-x86_64.sh

source ~/.bashrc