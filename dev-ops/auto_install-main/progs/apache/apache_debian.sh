#!/bin/bash

. ./general_start.sh

# It is recommended that you enable the most restrictive profile that will still allow the traffic you’ve configured. Since we haven’t configured SSL for our server yet in this guide, we will only need to allow traffic on port 80:

sudo ufw allow 'WWW'

. ./general_end.sh