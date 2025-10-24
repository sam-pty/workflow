# workflow

A concise README for this repository.

## Overview

This repository contains tooling and research code for DeepFlow and ASTRA-sim:
- DeepFlow: performance & energy modeling, architecture search, and validation scripts.
  See [DeepFlow/README.md](DeepFlow/README.md).
- ASTRA-sim: network and system simulator (analytical, ns-3, and garnet backends).
  See [astra-sim/README.md](astra-sim/README.md).

## Quickstart

1. Clone repo
```sh
git clone --recursive <repo>
```

2. Prepare Python environment:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tqdm
```

3. Run entire workflow
To generate execution traces (neccessary for first run), comment out below in `main` in `workflow.py`
```python
runner = DeepFlowRunner()
runner.run_all()

parser = DeepFlowParser()
parser.parse_all()
```
Then, run astra-sim, run ``run_all()`` ``for AstraSimRunner``, only ``install_chakra`` for first run to your venv, if see protobuf version error, use pip to force install a version it prompts.
```python
def run_all(self):
    self.compile_astrasim()
    self.install_chakra()
    self.convert_text_to_chakra()
    self.run_astrasim()
```
