# caffe2any: analyze Caffe prototxt files

This project contains a few utilities to analyze Caffe prototxt files.
My aim is to get a better understanding of the structure of DL topologies, either by extracting layer statistics or by generating visualizations of the network.

The scripts extract structural information from a protoxt network.  Hopefully the distilled view will give you a better understanding of the most important aspects of a network.  For example, I used it to understand  what layers are used, how many instances of each layer exist and how they are paramatized.  I also wanted to extract the exact data blob dimensions, as the data moves thru the layers.

The information and "printed" into one of several "printer" types:
- A 'console' printer emits to the console
- A 'csv' printer generates CSV files containing network statistics or layer-by-layer information
- A 'png' printer generates PNG files

The 'display' option selects what information you want printed:
- 'inventory': prints a table summarizing the layer types used in the network.  For example:
- 'unique': prints a table summarizing the uniquely paramaterized layers
- 'mem': prints memory usage information
- 'bfs': prints the layers of the network by performing a BFS traversal

To parse prototxt files, I'm using Google's protobufs, obviuosly.  Since I was interested in looking at Fast and Faster R-CNN netowrks, I used a branched caffe.proto file.
- The included caffe_pb2.py was generated using protoc:
  $ protoc -I=.  --python_out=. ./caffe.proto

Usage:

example(1):<br>
$ python prototxt2csv.py examples/alexnet.deploy.prototxt  -p console -d mem,unique,inventory,output,bfs

example(2):<br>
$ python prototxt2csv.py examples/googlenetv1.deploy.prototxt -p png -d bfs

- This generates a PNG file from the network file.  It differs from prototxt2png mainly thru customization, extra edge annotation, and the fact that BLOBs are not treated as nodes.
- File printers/png.py can be customized to control the generated PNG images.  Change 'options' and 'theme'.

prototxt2png: draw the prototxt networks without installing Caffe
- I plan to retire this file, since I've incorporated its functionality into the main program.
- This is motivated by the need to generate PNG files without installing Caffe.  It reuses the original Caffe code, with a few changes.

Installation and dependencies:
- See file DEPENDENCIES.md
