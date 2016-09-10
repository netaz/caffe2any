"""
Classes that print to different media
"""

# See - http://stackoverflow.com/questions/2970858/why-doesnt-print-work-in-a-lambda
from __future__ import print_function
from collections import Counter

pooling_type = { 0:'MAX', 1:'AVG', 2:'STOCHASTIC'}
lrn_type = { 0:'ACROSS_CHANNELS', 1:'WITHIN_CHANNEL'}
PIXEL_SIZE_BYTES = 2   # TODO: make this configrable

class BasePrinter:
	def count_nodes(self, tplgy):
		node_cnt = []
		tplgy.traverse(lambda node: node_cnt.append(node.type))
		return Counter(node_cnt)

class ConsolePrinter(BasePrinter):
	"""A simple console printer"""

	def print_pool(self, node):
		return 'Pooling', \
			   pooling_type[node.pool_type] + ', k='+str(node.kernel_size)+'x'+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad)
		
	def print_deconv(self, node):
		return 'Deconvolution', \
			   'k='+str(node.kernel_size)+"x"+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad)

	def print_conv(self, node):
		return 'Convolution', \
			   'k='+str(node.kernel_size)+"x"+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad)

	def print_lrn(self, node):
		return 'LRN', \
			   lrn_type[node.norm_region] + ' size=' + str(node.local_size) + ' alpha=' + str(node.alpha) + ' beta=' + str(node.beta)

	def print_unknown(self, node):
		return node.type, ""

	def print_layer(self, node, count):
	    print_fn =  {
	        "Pooling" : self.print_pool,
	        "Convolution" : self.print_conv,
	        "Deconvolution" : self.print_deconv,
	        "LRN" : self.print_lrn,
	    }.get(node.type, self.print_unknown)
	    row_format ="{:<20} {:<45} {:<40}"  # 3 is the number of cols
	    #print('\t%-20s%-3s Count=%-10d' % print_fn(layer, count))
	    print (row_format.format(*(print_fn(node) + (count,))))

	def print_inventory(self, tplgy):
		print ("Inventory:\n----------")
		node_types_cnt = self.count_nodes(tplgy)
		for type in node_types_cnt:
			print('\t%-20s%-3i' % (type, node_types_cnt[type] ))
		print("Total=", len(tplgy.nodes))
		print("")

	def print_unique(self, unique_layers_list):
		for node in unique_layers_list:
			self.print_layer(node[0], node[1]) # print node, count

	def print_unique_all(self, unique_layers_dict):
		print ("Unique:\n--------")
		for type_name in unique_layers_dict:
			self.print_unique(unique_layers_dict[type_name])
		print("")

	def print_bfs(self, tplgy):
		tplgy.traverse(lambda node: print(str(node)), 
			 		   lambda edge: print('\t' + str(edge)))

class CsvPrinter(BasePrinter):
	"""A CSV file printer"""

	def __init__(self, fname):
		self.file = open(fname, "wt")

	def print_pool(self, node):
		return "Pool," + pooling_type[node.pool_type] + ' k='+str(node.kernel_size)+"x"+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad)
		
	def print_deconv(self, node):
		return 'Deconvolution, k='+str(node.kernel_size)+"x"+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad) 

	def print_conv(self, node):
		return 'Convolution, k='+str(node.kernel_size)+"x"+str(node.kernel_size) + '/s='+str(node.stride) + ' pad='+str(node.pad)

	def print_lrn(self, node):
		return 'LRN,' + lrn_type[node.norm_region] + ' local_size=' + str(node.local_size) + ' alpha=' + str(node.alpha) + ' beta=' + str(node.beta)

	def print_unknown(self, node):
		return str(node.type) + ','
	
	def print_layer(self, node):
	    print_fn =  {
	        "Pooling" : self.print_pool,
	        "Convolution" : self.print_conv,
	        "Deconvolution" : self.print_deconv,
	        "LRN" : self.print_lrn,
	    }.get(node.type, self.print_unknown)
	    return print_fn(node)

	def print_MACs(self, node, ofms_descriptor, tplgy):
		if node.type != 'Convolution':
			return '0'

		edges = tplgy.find_incoming_edges(node)
		assert(len(edges) == 1)

		num_ifms = edges[0].blob.shape[1]
		
		# macs = #OFMs*OFM_X*OFM_Y*#IFMs*K_X*K_Y
		num_ofms = ofms_descriptor[1]
		ofm_x = ofms_descriptor[2]
		ofm_y = ofms_descriptor[3]
		MACs = num_ofms * ofm_x * ofm_y * num_ifms * node.kernel_size * node.kernel_size
		return str(MACs)


	def print_inventory(self, tplgy):
		node_types_cnt = self.count_nodes(tplgy)

		self.file.write('Type, Count\n')
		for type in node_types_cnt:
			line = type + ',' + str(node_types_cnt[type]) + '\n'
			self.file.write(line)
		self.file.write('\n')

	def print_unique(self, unique_layers_list):
		for node in unique_layers_list:
			self.file.write(self.print_layer(node[0]) + '\n')

	def print_unique_all(self, unique_layers_dict):
		self.file.write('Type, Configuration\n')
		for type_name in unique_layers_dict:
			self.print_unique(unique_layers_dict[type_name])
		self.file.write('\n')

	def print_bfs(self, tplgy):
		self.file.write('Node, Type, Details,OFMz,OFMy,OFMx,Size (pixels), Size (bytes), MACs\n')
		self.done_blobs = []
		tplgy.traverse(None, lambda edge: self.print_edge_cb(edge, tplgy))

	def print_edge_cb(self, edge, tplgy):
		if edge.blob in self.done_blobs:
			return # been there, done that

		self.done_blobs.append(edge.blob)
		size = 0
		if edge.blob.shape and edge.src_node.role!="Modifier":
			size = edge.blob.size()
			
		self.file.write(
			(edge.src_node.name if edge.src_node else '') + ',' +						# Node name
			(str(self.print_layer(edge.src_node)) if edge.src_node else ',') +  ',' +	# Layer type, details
			(str(edge.blob.shape[1]) if edge.blob.shape else '') + ',' +				# OFMz
			(str(edge.blob.shape[2]) if edge.blob.shape else '')+ ',' +					# OFMy
			(str(edge.blob.shape[3]) if edge.blob.shape else '') + ',' +				# OFMx
			str(size) + ',' +															# size - pixels
			str(size*PIXEL_SIZE_BYTES) + ',' +											# size - bytes
			self.print_MACs(edge.src_node, edge.blob.shape, tplgy) + ',' +				# MACs
			'\n')
