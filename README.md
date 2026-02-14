# vae-project
![](/notebooks/langevin.gif)

Project on Variational Autoencoders (VAEs) - a machine learning technique for learning compressed representations of data. This project includes implementations for Colored MNIST dataset.

## Quick Start

### 1. Get the code

Download this project to your computer:

```bash
git clone https://github.com/ssslakter/vae-project.git
cd vae-project
```

### 2. Set up the environment

You need to install dependencies. Choose one method:

#### Option A: Using Pixi (recommended, easiest)

Pixi is similar to conda and automatically manages all dependencies for you:

```bash
pixi install
pixi shell -e dev # will also install jupyter into venv
```

#### Option B: Using pip and virtual environment

Create an isolated Python environment and install dependencies:

```bash
python -m venv venv          # Create isolated environment
source venv/bin/activate     # Activate it
pip install .                # Install the project
```

## Project Structure

- `notebooks/` - Interactive Jupyter notebooks to run experiments
- `vae_project/` - Main code package
  - `models/` - The VAE neural network architecture
  - `dataset/` - Code to load and prepare datasets
  - `train/` - Training scripts and monitoring tools
- `data/` - Contains the image datasets used for training

## On commits in notebooks
If you want to make things easier when updating notebooks and make sure you don't commit your metadata (cell execution num, timestamps, etc.) changes, you can use cli tool called `nbdev`.

To clean metadata in all notebooks, run (it will keep all outputs, cells and you can continue to use the notebook):
```bash
nbdev_clean --fname .
```

It's already in optional dependencies and you can use it inside pixi shell. Or you can also install it with `pip install nbdev`.
