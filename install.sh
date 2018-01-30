#!/bin/bash

pip install -r requirements.txt

mkdir deps
cd deps

git clone -b dev_jeff https://github.com/BerkeleyAutomation/perception.git
cd perception
python setup.py develop
cd ..

cd ..
python setup.py develop
