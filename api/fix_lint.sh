#!/usr/bin/env bash
black .
isort -y
flake8 --select=F --config .config/flake8