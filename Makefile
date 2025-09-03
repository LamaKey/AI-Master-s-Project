SHELL := /bin/sh

PROJECT_NAME := ai-fairness-hiring
IMAGE := $(PROJECT_NAME):latest

.PHONY: help
help:
	@echo "Targets:"
	@echo "  venv           - create Python venv (./venv)"
	@echo "  install        - install dependencies into active environment"
	@echo "  run            - run the pipeline (python main.py)"
	@echo "  clean          - remove results/ outputs"
	@echo "  docker-build   - build Docker image ($(IMAGE))"
	@echo "  docker-run     - run pipeline in Docker (writes results/)"
	@echo "  docker-shell   - open shell inside image"

.PHONY: venv
venv:
	python -m venv venv

.PHONY: activate
activate:
	.\venv\Scripts\activate

.PHONY: install
install:
	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: run
run:
	python main.py

.PHONY: clean
clean:
	rm -rf results
	mkdir -p results/plots results/reports

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm \
	  -v $(PWD)/results:/app/results \
	  -v $(PWD)/data:/app/data \
	  $(IMAGE)

.PHONY: docker-shell
docker-shell:
	docker run --rm -it \
	  -v $(PWD)/results:/app/results \
	  -v $(PWD)/data:/app/data \
	  $(IMAGE) /bin/bash


