This file contains instructions for installing this project's external dependencies.
I'm using Ubuntu 14.04.1.

1. Install python 3.x
   I'm using  Anaconda.  You can find the official instruction here: https://docs.continuum.io/anaconda/install/linux.html

2. Install protobufs for Python. If you're using Anaconda, install like so:
   > conda install -c anaconda protobuf

3. Install pydot.  pydot is not supported under python 3 and pydot2 doesn't work properly.
   However, pydotplus works nicely.  To install with Anaconda:
   > conda install -c conda-forge pydotplus 
     
   To install with pip:
   > pip install pydotplus

4. Install graphviz
   > sudo apt-get install graphviz
