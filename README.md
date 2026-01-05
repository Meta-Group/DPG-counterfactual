# CounterFactualDPG

For Linux/Mac Users:
  ```bash
  # Create a virtual environment
  python -m venv .venv

  # Activate the virtual environment
  source .venv/bin/activate

  # Install DPG
  pip install -r ./requirements.txt
  ```
nbstripout as a pre-commit hook is recommended to keep notebooks clean of output cells.
```
nbstripout --install
```