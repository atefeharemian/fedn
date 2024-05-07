#!/bin/bash
set -e

# Init venv
# python -m venv .IOT

# Pip deps
.IOT/bin/pip install --upgrade pip
.IOT/bin/pip install -e ../../fedn
.IOT/bin/pip install -r requirements.txt
