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
from google.protobuf import text_format

pooling_type = { 0:'MAX', 1:'AVG', 2:'STOCHASTIC'}
lrn_type = { 0:'ACROSS_CHANNELS', 1:'WITHIN_CHANNEL'}

def print_pool(layer):
	pool = layer.pooling_param.pool
	kernel_size = layer.pooling_param.kernel_size
	stride = layer.pooling_param.stride
	pad = layer.pooling_param.pad
	return "[Pool="+pooling_type[pool] + '] k='+str(kernel_size)+"x"+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad)

def print_deconv(layer):
	return layer.convolution_param

def print_conv(layer):
	param = layer.convolution_param
	kernel_size = 1 if len(param.kernel_size)==0 else param.kernel_size[0] #  defaults to 1
	stride = 1 if len(param.stride)==0 else param.stride[0] # The stride; defaults to 1
	pad = 0 if len(param.pad)==0 else param.pad[0] # The padding size; defaults to 0
	return '[Conv] k='+str(kernel_size)+"x"+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad)

def print_lrn(layer):
	param = layer.lrn_param
	local_size = layer.lrn_param.local_size # default = 5
	alpha = layer.lrn_param.alpha # default = 1.
	beta = layer.lrn_param.beta # default = 0.75
	type = layer.lrn_param.norm_region
	return '[LRN='+ lrn_type[type] + '] local_size=' + str(local_size) + ' alpha=' + str(alpha) + ' beta=' + str(beta)

def print_unknown(layer):
#	print "UNKNOWN"
	pass

def print_layer(layer):
    print_fn =  {
        "Pooling" : print_pool,
        "Convolution" : print_conv,
        "Deconvolution" : print_deconv,
        "LRN" : print_lrn,
    }.get(layer.type, print_unknown)
    print "\t" + print_fn(layer)

def print_summary(layer_types):
	print "Summary:"
	print "--------"
	for type in layer_types:
		print "\t" + type, layer_types[type] 

def print_unique(unique_layers_list):
	for layer in unique_layers_list:
		print_layer(layer)

def print_unique_all(unique_layers_dict):
	print "Unique:"
	print "--------"
	for type_name in unique_layers_dict:
		print_unique(unique_layers_dict[type_name])

def is_equal_conv(layer1, layer2):
	param1 = layer1.convolution_param
	kernel_size1 = param1.kernel_size[0]
	stride1 = 1 if len(param1.stride)==0 else param1.stride[0] 
	pad1 = 0 if len(param1.pad)==0 else param1.pad[0] 

	param2 = layer2.convolution_param
	kernel_size2 = param2.kernel_size[0]
	stride2 = 1 if len(param2.stride)==0 else param2.stride[0] 
	pad2 = 0 if len(param2.pad)==0 else param2.pad[0] 

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
	net = caffe_pb2.NetParameter()
	
	# Read a Caffe prototxt file
	try:
	  f = open(sys.argv[1], "rb")
	  text_format.Parse(f.read(), net)
	  f.close()
	except IOError:
	  print sys.argv[1] + ": Could not open file.  Creating a new one."

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

	print_summary(layer_types)
	#print_unique(unique_layers["Pooling"])
	#print_unique(unique_layers["Convolution"])
	print_unique_all(unique_layers)

if __name__ == '__main__':
    main()