.PHONY: data train index eval rank test all

data:
	python3 src/data_preprocess.py

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

all: data train index eval rank test
