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
