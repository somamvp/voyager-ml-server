# voyager-ml-server

## Requirements

- Ubuntu (20.04 tested)
- conda-forge

## Getting Started

```shell
# clone giithub repo, including submodules
git clone --recurse-submodules https://github.com/somamvp/voyager-ml-server.git
cd voyager-ml-server

# import conda env: 'server-gpu-env'
conda env create -f conda_environments.yaml
conda activate server-gpu-env

# run server script
python server.py
