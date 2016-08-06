#!/usr/bin/python
	
"""
Start by downloading the Caffe proto file from 
https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

Then use protoc to compile it to caffe_pb2
protoc caffe.proto

Finally use to parse it

summary
 -l [deconv|conv|]
"""

import caffe_pb2
import sys
import argparse
from google.protobuf import text_format
from printers import ConsolePrinter, CsvPrinter


def is_equal_conv(layer1, layer2):
	param1 = layer1.convolution_param
	kernel_size1 = param1.kernel_size
	stride1 = param1.stride
	pad1 = param1.pad
	
	param2 = layer2.convolution_param
	kernel_size2 = param2.kernel_size
	stride2 = param2.stride
	pad2 = param2.pad

	return (kernel_size1 == kernel_size2 and stride1 == stride2 and pad1==pad2)

def is_equal(layer1, layer2):
	assert layer1.type == layer2.type
	if layer1.type == "Pooling": 
		return layer1.pooling_param == layer2.pooling_param
	if layer1.type == "Convolution": 
		return is_equal_conv(layer1, layer2)
	if layer1.type == "LRN": 
		return layer1.lrn_param == layer2.lrn_param
	return True

def is_unique(layer, unique_list):
	unique = True
	for unique_layer in unique_list:
		if is_equal(unique_layer, layer):
			unique = False
	return unique	

def add_unique(layer, unique_layers):
	if unique_layers.get(layer.type)==None:
		unique_layers[layer.type] = []
	if is_unique(layer, unique_layers[layer.type]):
		unique_layers[layer.type].append(layer)


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('infile', help='input prototxt file')
	parser.add_argument('-f', '--format', help='output format (csv, console)', default='console')
	args = parser.parse_args()

	net = caffe_pb2.NetParameter()
	
	# Read a Caffe prototxt file
	try:
		f = open(sys.argv[1], "rb")
	  	text_format.Parse(f.read(), net)
	  	f.close()
	except IOError:
	  	print "Could not open file ", sys.argv[1]

	layer_types = {}
	unique_layers = {}
	
	for layer in net.layer:
		#print layer.name
		if layer.type == "Convolution":
			add_unique(layer, unique_layers)

		if layer.type == "Pooling":
			add_unique(layer, unique_layers)

		if layer.type == "LRN":
			add_unique(layer, unique_layers)

		if layer_types.get(layer.type) == None:
			layer_types[layer.type] = 1
		else:
			layer_types[layer.type] = layer_types[layer.type] + 1

	if args.format == 'console':
		printer = ConsolePrinter()
	else:
		printer = CsvPrinter(args.infile + '.csv') 
	printer.print_summary(layer_types)
	#print_unique(unique_layers["Pooling"])
	#print_unique(unique_layers["Convolution"])
	printer.print_unique_all(unique_layers)

if __name__ == '__main__':
    main()