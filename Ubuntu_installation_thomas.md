# create and activate virtual python environment
python3 -m venv <name>
  
source <name>/bin/activate

# install python packages
pip install jupyterlab

pip install numpy

pip install h5py

pip install opencv-contrib-python

_(if there is an error first: sudo apt install cmake && pip install scikit-build)_

pip install joblib

pip install matplotlib

pip install ipympl

pip install sidecar

pip install scipy

pip install -U scikit-learn==0.20.1

pip install sklearn

# installing jupyter widgets
sudo apt install npm

sudo npm install npm@latest -g

sudo npm cache clean -f

sudo npm install -g n

sudo n stable

sudo apt-get update

sudo apt-get install build-essential checkinstall libssl-dev

curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.32.1/install.sh | bash

jupyter labextension install @jupyter-widgets/jupyterlab-manager

jupyter lab build

jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib@0.7.4

# things to change in code
experiment_path

tracks_path
