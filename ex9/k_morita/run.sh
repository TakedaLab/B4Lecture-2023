#!/bin/bash

source .venv/bin/activate

echo ***** Train *****
python -u main.py


echo ***** Evaluate *****
python eval.py

