#!/bin/bash
sudo apt install python3 python3-venv python3-pip
python3 -m venv ./OllamaWebPDF_env
source ./OllamaWebPDF_env/bin/activate
pip install -r ./requirements.txt

