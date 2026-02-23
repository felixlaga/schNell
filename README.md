# schNell
schNell is a very lightweight python module that can be used to compute basic map-level noise properties for generic networks of gravitational wave interferometers. This includes primarily the noise power spectrum  "N_ell", but also other things, such as antenna patterns, overlap functions, inverse variance maps etc.

## Installation

### NEW installation instructions
Use `uv` as a package manager. This is faster, more modern than conda or pip. All project dependencies are summarized in the pyproject.toml file. This is used to build the package. The uv.lock contains everuthing to create the environment. 

Install uv:
```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment:
```console
uv sync 
```
This will install the dependencies and create a virtual environment. 
Now we launch the environment and run jupyter:
```console
# activate env
source .venv/bin/activate

# install schNell
uv pip install .

# launch local jupyter server
uv run --with jupyter jupyter lab
```

### Legacy
You can install schnell simply by typing
```
pip install schnell
```
(use `--user` if you don't have admin privileges on your machine).
Or for development versions you can download the repository with git and install from there using `python setup.py install [--user]`.

## Documentation
Documentation can be found on [readthedocs](https://schnell.readthedocs.io/en/latest/).

This example [notebook](https://github.com/damonge/schNell/blob/master/examples/Nell_example.ipynb) on github also showcases the main functionalities of the module.

Check out the following videos showing the scanning patterns of different GW networks.
- [LIGO instantaneous sensitivity](https://youtu.be/54WBdWgBO8k)
- [LIGO cumulative sensitivity](https://youtu.be/ByrEqpIrQzY)
- [LISA instantaneous sensitivity](https://youtu.be/8d6gEGlboz8)

## License and credits
If you use schNell, we kindly ask you to cite its [companion paper](https://arxiv.org/abs/2005.03001).

The code is available under the [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.

If you have a problem you've not been able to debug, or a feature request/suggestion, please open an issue on github to discuss it.
