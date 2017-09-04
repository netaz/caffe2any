This file contains instructions for installing this project's external dependencies.
I'm using Ubuntu 14.04.1.

1. Install python 3.x<br>
   I'm using  Anaconda.  You can find the official instruction here: https://docs.continuum.io/anaconda/install/linux.html
1. Create and activate the Anaconda environment<br> 
`conda env create -f environment.yml`<br>
`source activate caffe2any`
1. Install graphviz<br>
`sudo apt-get install graphviz`

<br>
<br>
Here's how I created environment.yaml:
<br>
>  conda create -n caffe2any python=3.5
This creates an environment in /home/nzmora/data/anaconda3/envs/caffe2any

> conda install -c conda-forge tensorflow
This installs TF in the environment
To install TF on Anaconda see: https://www.tensorflow.org/versions/r0.12/get_started/os_setup#anaconda_installation

> conda install -c conda-forge matplotlib, pyaml
This installs matplotlib, yaml

> conda install -c conda-forge pydotplus
This installs pydot.  pydot is not supported under python 3 and pydot2 doesn't work properly.
However, pydotplus works nicely.  To install with Anaconda (https://anaconda.org/conda-forge/pydotplus):
<br>
You can find more information here:  http://www.installion.co.uk/ubuntu/precise/universe/p/python-pydot/install/index.html

> conda install -c anaconda protobuf
<br>
Finally, create the YAML file:
> conda env export > environment.yml
