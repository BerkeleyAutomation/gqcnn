#!/bin/bash

pip install -r requirements.txt

mkdir deps
cd deps

git clone https://github.com/BerkeleyAutomation/perception.git
cd perception
python setup.py develop
cd ..

cd ..
python setup.py develop
