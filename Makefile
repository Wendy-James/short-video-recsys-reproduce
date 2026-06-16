.PHONY: data negatives train index eval rank test all run_all

data:
	python3 src/data_preprocess.py

negatives:
	python3 src/negative_sampling.py

train:
	python3 src/train_twotower.py

index:
	python3 src/build_faiss_index.py

eval:
	python3 src/evaluate_recall.py

rank:
	python3 src/train_ranker.py

test:
	python3 -m pytest -q

run_all:
	python3 run_all.py

all: data negatives train index eval rank test
	cp outputs/metrics.csv experiments/generated_metrics.csv
