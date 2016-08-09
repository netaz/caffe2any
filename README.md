# prototxt2csv
a few utilities to analyze Caffe prototxt files

prototxt2csv: analyze prototxt files
- This tries to extract some structural information from the protoxt network.  Hopefully the distilled view will give you a better understanding of the most important aspects of a network.  For example, I used it to understand  what layers are used, how many instances of each layer exist and how they are paramatized.  I also wanted to extract the exact data blob dimensions, as the data moves thru the layers.
- To parse the prototxt files, I'm using Google's protobufs, obviuosly.  Since I was interested in looking at Fast and Faster R-CNN netowrks, I used a branched caffe.proto file.
- The included caffe_pb2.py was generated using protoc:
  $ protoc -I=.  --python_out=. ./caffe.proto

prototxt2png: draw the prototxt networks without installing Caffe
- This is motivated by the need to generate PNG files without installing Caffe.  It reuses the original Caffe code, with a few changes.
- You will to install pydot (instructions: http://www.installion.co.uk/ubuntu/precise/universe/p/python-pydot/install/index.html)

