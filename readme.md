# 5 day generative AI

## Installation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Usage

source /nvme0/scott/venv/chess/bin/activate

Generate 100k data:

./encoder/generate_data.py data/last-100-k-lichess_db_standard_rated_2023-05.pgn  -o data/out-100k.csv -c 100000

./encoder/generate_data.py ../../private/work-in-progress/chess/data/first-1-5-mil_lichess_db_standard_rated_2023-05.pgn data/out-1-mil.csv -o -c 1000000

Train:

./encoder/train.py data/out-1-mil.csv

Test:

./encoder/test.py data/out-100k.csv

## Experiments

v1.0.0: First simple CNN, train on 1 mil, test on 100k

* early exit, somewhere around epoch 20
* Test Loss: 0.0552
* Test Accuracy: 0.9785

v2.0.0: Change model, adding BatchNormalization and Activation after each Conv2D

* early exit, best epoch 30
* Test Loss: 0.0254
* Test Accuracy: 0.9922

v2.0.1: Change data, random mainline position instead of end position

*
