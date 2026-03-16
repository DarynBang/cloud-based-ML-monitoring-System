# Makefile for Python project on Windows

VENV = .venv
PYTHON = $(VENV)\Scripts\python
PIP = $(VENV)\Scripts\pip

.PHONY: all venv install clean activate

all: venv install

venv:
	python -m venv $(VENV)
	@echo Virtual environment created in $(VENV)

install: venv
	$(PIP) install --upgrade pip
	@if exist requirements.txt ( \
		$(PIP) install -r requirements.txt \
	) else ( \
		echo "No requirements.txt found, skipping installation." \
	)

activate:
	$(VENV)\Scripts\activate.bat

clean:
	@if exist $(VENV) rmdir /s /q $(VENV)
	@echo Virtual environment removed.
