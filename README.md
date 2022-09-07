# voyager-ml-server

## Requirements

- Ubuntu (20.04 tested)
- conda-forge

## Getting Started

```shell
# clone giithub repo, including submodules
git clone --recurse-submodules https://github.com/somamvp/voyager-ml-server.git
cd voyager-ml-server

# import conda env: 'yolov7-env'
conda env create -f conda_environments_yolov7.yaml
conda activate yolov7-env

# run server script
run_fastapi.sh
