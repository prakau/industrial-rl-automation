# Python Test Project

## Project Structure
```
python-test-project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── tests/
│   └── test_example.py
├── src/
│   └── example.py
├── requirements.txt
└── README.md
```

## GitHub Actions Workflow

## License: MIT & Unknown Sources
Multiple sources were used to create this GitHub Actions workflow:
- https://github.com/dongjun1217/dongjun1217.github.io (MIT License)
- https://github.com/falkievich/Tarea-de-Ing.-de-Software-II-2023 (Unknown License)
- https://github.com/Cherrue/Cherrue.github.io (Unknown License)

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest

