!!! note

    Group name: Alpha24

## Usage

### Using UV package manager

!!! note

    In order to use UVs package manager, please make sure to install it. You can
    find installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).
    In case you are using Linux/MacOS, you can run the following command:

    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

Once you have cloned the repository, you can run `uv sync` in order to download any package that you might not
have. In case you need to add a new package, use `uv add name_of_pkg`.

To run a python script with the isolated environment you have 2 ways.

1. Run the script with `uv run name_of_script.py` which comes to the same as `python3 name_of_script.py`.
   The only difference is that you have the environment loaded when using uv.

2. Create a `.venv`. Run the `uv venv` which will generate a `.venv`. Then just activate is using `source .venv/bin/activate` and then
   use it normally as any other python virtual environment.

### Conda

!!! note

    Conda can be installed as a minimal package by following the guide [here](https://docs.anaconda.com/miniconda/).

Upon cloning the repository, run `conda env create -f environment.yml` while inside the repository. Once the environment is set up, activate it with `conda activate spark`.
