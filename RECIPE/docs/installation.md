# Installation

## Requirements

- Python `>=3.10`
- A working PyTorch installation
- `torch-geometric` compatible with your PyTorch build

## Install From GitHub

Because the packaged Python project is stored in the `RECIPE/` subdirectory of the repository, install it with `subdirectory=RECIPE`:

```bash
python -m pip install -U pip
python -m pip install "git+https://github.com/mcgilldinglab/RECIPE.git@main#subdirectory=RECIPE"
```

## Editable Install From A Clone

```bash
git clone https://github.com/mcgilldinglab/RECIPE.git
cd RECIPE/RECIPE
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verify The Install

```bash
python -c "import recipe; print(recipe.__file__)"
```

## Build Documentation Locally

```bash
cd RECIPE/RECIPE
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```
