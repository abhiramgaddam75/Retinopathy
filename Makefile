# DR-XAI pipeline targets (DESIGN.md Section 6). On Windows without `make`,
# run the `python -m xaidr.cli ...` commands from README.md directly.
PY ?= python
CONFIG ?= configs/default.yaml
RUN ?=

.PHONY: setup prepare splits train evaluate xai crossdataset ablation efficiency figures test reproduce

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

prepare:
	$(PY) -m xaidr.cli prepare-data --config $(CONFIG)
	$(PY) -m xaidr.cli make-splits --config $(CONFIG)

train:
	$(PY) -m xaidr.cli train --config $(CONFIG)

evaluate:
	$(PY) -m xaidr.cli evaluate --run $(RUN)

xai:
	$(PY) -m xaidr.cli xai --run $(RUN)

crossdataset:
	$(PY) -m xaidr.cli crossdataset --run $(RUN)

ablation:
	$(PY) -m xaidr.cli ablation --config configs/ablation_no_oversampling.yaml

efficiency:
	$(PY) -m xaidr.cli efficiency --run $(RUN)

figures:
	$(PY) -m xaidr.cli figures --run $(RUN)

test:
	$(PY) -m pytest tests

reproduce:
	$(PY) -m xaidr.cli all --config $(CONFIG)
	$(PY) -m pytest tests
