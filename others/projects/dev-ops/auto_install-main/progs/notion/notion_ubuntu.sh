#!/bin/bash

current_program_name="notion"

echo "deb [trusted=yes] https://apt.fury.io/notion-repackaged/ /" | sudo tee /etc/apt/sources.list.d/notion-repackaged.list

sudo apt update

sudo apt install notion-app-enhanced

sudo apt install notion-app