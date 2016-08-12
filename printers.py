"""
Classes that print to different media
"""

# See - http://stackoverflow.com/questions/2970858/why-doesnt-print-work-in-a-lambda
from __future__ import print_function
from collections import Counter

pooling_type = { 0:'MAX', 1:'AVG', 2:'STOCHASTIC'}
lrn_type = { 0:'ACROSS_CHANNELS', 1:'WITHIN_CHANNEL'}

class BasePrinter:
	def print_pool(self, layer):
		pool = layer.pooling_param.pool
		kernel_size = layer.pooling_param.kernel_size
		stride = layer.pooling_param.stride
		pad = layer.pooling_param.pad
		return pool, kernel_size, stride, pad

	def print_conv(self, layer):
		param = layer.convolution_param
		kernel_size = param.kernel_size #  defaults to 1
		stride = param.stride # The stride; defaults to 1
		pad = param.pad # The padding size; defaults to 0
		return kernel_size, stride, pad

	def print_lrn(self, layer):
		param = layer.lrn_param
		local_size = layer.lrn_param.local_size # default = 5
		alpha = layer.lrn_param.alpha # default = 1.
		beta = layer.lrn_param.beta # default = 0.75
		type = layer.lrn_param.norm_region
		return local_size, alpha, beta, type

	def print_layer(self, layer):
	    print_fn =  {
	        "Pooling" : self.print_pool,
	        "Convolution" : self.print_conv,
	        "Deconvolution" : self.print_deconv,
	        "LRN" : self.print_lrn,
	    }.get(layer.type, self.print_unknown)

	def count_nodes(self, tplgy):
		node_cnt = []
		tplgy.traverse(lambda node: node_cnt.append(node.type))
		return Counter(node_cnt)

class ConsolePrinter(BasePrinter):
	"""A simple console printer"""

	def print_pool(self, layer):
		pool, kernel_size, stride, pad = BasePrinter.print_pool(self, layer)
		return 'Pool='+pooling_type[pool], \
			   'k='+str(kernel_size)+'x'+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad)
		
	def print_deconv(self, layer):
		return layer.convolution_param

	def print_conv(self, layer):
		kernel_size, stride, pad = BasePrinter.print_conv(self, layer)
		return 'Convolution', \
			   'k='+str(kernel_size)+"x"+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad)

	def print_lrn(self, layer):
		local_size, alpha, beta, type = BasePrinter.print_lrn(self, layer)
		return 'LRN='+ lrn_type[type], \
			   'local_size=' + str(local_size) + ' alpha=' + str(alpha) + ' beta=' + str(beta)

	def print_unknown(self, layer):
		return layer.type, ""

	def print_layer(self, layer):
	    print_fn =  {
	        "Pooling" : self.print_pool,
	        "Convolution" : self.print_conv,
	        "Deconvolution" : self.print_deconv,
	        "LRN" : self.print_lrn,
	    }.get(layer.type, self.print_unknown)
	    print('\t%-20s%-3s' % print_fn(layer))

	def print_catalog(self, tplgy):
		print ("Catalog:\n--------")
		node_types_cnt = self.count_nodes(tplgy)
		for type in node_types_cnt:
			print('\t%-20s%-3i' % (type, node_types_cnt[type] ))

	def print_unique(self, unique_layers_list):
		for layer in unique_layers_list:
			self.print_layer(layer)

	def print_unique_all(self, unique_layers_dict):
		print ("Unique:\n--------")
		for type_name in unique_layers_dict:
			self.print_unique(unique_layers_dict[type_name])

	def print_bfs(self, tplgy):
		tplgy.traverse(lambda node: print(str(node)), 
			 		   lambda edge: print('\t' + str(edge)))

class CsvPrinter(BasePrinter):
	"""A CSV file printer"""

	def __init__(self, fname):
		self.file = open(fname, "wt")

	def print_pool(self, layer):
		pool, kernel_size, stride, pad = BasePrinter.print_pool(self, layer)
		return "Pool," + pooling_type[pool] + ' k='+str(kernel_size)+"x"+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad)
		
	def print_deconv(self, layer):
		return layer.convolution_param

	def print_conv(self, layer):
		kernel_size, stride, pad = BasePrinter.print_conv(self, layer)
		return 'Convolution, k='+str(kernel_size)+"x"+str(kernel_size) + '/s='+str(stride) + ' pad='+str(pad) 

	def print_lrn(self, layer):
		local_size, alpha, beta, type = BasePrinter.print_lrn(self, layer)
		return 'LRN,' + lrn_type[type] + 'local_size=' + str(local_size) + ' alpha=' + str(alpha) + ' beta=' + str(beta)

	def print_unknown(self, layer):
		return layer.type + ','
		
	def print_layer(self, layer):
	    print_fn =  {
	        "Pooling" : self.print_pool,
	        "Convolution" : self.print_conv,
	        "Deconvolution" : self.print_deconv,
	        "LRN" : self.print_lrn,
	    }.get(layer.type, self.print_unknown)
	    self.file.write(print_fn(layer) + '\n')

	def print_catalog(self, tplgy):
		node_types_cnt = self.count_nodes(tplgy)

		self.file.write('Type, Count\n')
		for type in node_types_cnt:
			line = type + ',' + str(node_types_cnt[type]) + '\n'
			self.file.write(line)
		self.file.write('\n')

	def print_unique(self, unique_layers_list):
		for layer in unique_layers_list:
			self.print_layer(layer)

	def print_unique_all(self, unique_layers_dict):
		#unique_nodes = {}
		#tplgy.traverse(lambda node: add_unique(node.layer, unique_nodes))

		self.file.write('Type, Configuration\n')
		for type_name in unique_layers_dict:
			self.print_unique(unique_layers_dict[type_name])

	def print_bfs(self, tplgy):
		self.file.write('Node, Type, Details,OFMx,OFMy,OFMz\n')

		tplgy.traverse(None, 
			 		   lambda edge: self.file.write((edge.src_node.name if edge.src_node else '') + ',' +
			 		   								(edge.src_node.type if edge.src_node else '') +  ',' +
			 		   								',' +
			 		   								(str(edge.blob.shape[1]) if edge.blob.shape else '') + ',' +
			 		   								(str(edge.blob.shape[2]) if edge.blob.shape else '')+ ',' +
			 		   								(str(edge.blob.shape[3]) if edge.blob.shape else '') + ',' +
			 		   								'\n'))

#		        return ('Edge [' + str(self.blob) +  ': ' + (self.src_node.name if self.src_node else 'None') + ' ==> ' + 
 #               (self.dst_node.name if self.dst_node else 'None') +  ']')
