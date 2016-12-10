from globals import *
pooling_type = get_pooling_types_dict()

class CsvPrinter:
    """A CSV file printer"""

    def __init__(self, fname):
        self.file = open(fname, "wt")
        self.done_blobs = []
        # TODO - make this constants so they can be reused
        self.cols = ['Node', 'Type', 'Node Details', 'IFMz', 'IFMy', 'IFMx', 'OFMz', 'OFMy', 'OFMx',
                     'IFM Volume (elems)', 'OFM Volume (elems)', 'Weights Volume (elems)', 'Bias Volume (elems)', 'MACs']

    def print_pool(self, node):
        desc = "Pool," + pooling_type[node.pool_type] + ' k=' + str(node.kernel_size) + "x" + str(
            node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
        if node.ceiling:
            desc += ' ceiling'
        return desc

    def print_deconv(self, node):
        return 'Deconvolution, k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(
            node.stride) + ' pad=' + str(node.pad)

    def print_conv(self, node):
        return 'Convolution, k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(
            node.stride) + ' pad=' + str(node.pad)

    def print_conv_relu(self, merged_node):
        node = merged_node.node1
        return 'Convolution/ReLU, k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(
            node.stride) + ' pad=' + str(node.pad)

    def print_lrn(self, node):
        return 'LRN,' + lrn_type[node.norm_region] + ' local_size=' + str(node.local_size) + ' alpha=' + str(
            node.alpha) + ' beta=' + str(node.beta)

    def print_eltwise(selfself, node):
        op_lookup = get_eltwise_op_dict()
        return 'Eltwise,' + op_lookup[node.operation]

    def print_unknown(self, node):
        return str(node.type) + ','

    # Print extra node details (e.g. kernel size, when applicable)
    def print_node(self, node):
        print_fn = {
            "Pooling": self.print_pool,
            "Convolution": self.print_conv,
            "Convolution_ReLU": self.print_conv_relu,
            "Deconvolution": self.print_deconv,
            "LRN": self.print_lrn,
            "Eltwise": self.print_eltwise,
        }.get(node.get_type(), self.print_unknown)
        return print_fn(node)

    def print_ifms(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Convolution_ReLU', 'InnerProduct', 'InnerProduct_ReLU', 'Pooling', 'Deconvolution', 'Eltwise', 'LRN', 'Softmax']:
            ifm_shape = edges[0].blob.shape
            if ifm_shape is None:
                #print("node " + node.name + " has no ifm_shape")
                return ',,'
            return str(ifm_shape[1]) + ',' + str(ifm_shape[2]) + ',' + str(ifm_shape[3])
        else:
            return ',,'

    def get_weight_size(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Convolution_ReLU']:
            assert (len(edges) == 1)
            num_ifms = edges[0].blob.shape[1]
            if node.type == 'Convolution_ReLU':
                node = node.node1
            return node.kernel_size * node.kernel_size * node.num_output * num_ifms
        elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
            assert (len(edges) == 1)
            if node.type == 'InnerProduct_ReLU':
                return (self.get_ifm_size(node, tplgy) * node.node1.num_output)
            else:
                return (self.get_ifm_size(node, tplgy) * node.num_output)
        else:
            return 0

    # todo: move this to Topology (per Node class)
    def get_bias_size(self, node, tplgy):
        if node.type in ['Convolution', 'InnerProduct']:
            return node.num_output
        if node.type in ['Convolution_ReLU', 'InnerProduct_ReLU']:
            return node.node1.num_output
        else:
            return 0

    # todo: move this to Topology (per Node class)
    def get_ifm_size(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Convolution_ReLU', 'Pooling', 'LRN', 'Softmax']:
            assert (len(edges) == 1)
            ifm = edges[0].blob
            return ifm.size()
        elif node.type in ['Eltwise']:
            # Eltwise has two inputs of equal dimensions
            assert (len(edges) == 2)
            ifm = edges[0].blob
            return ifm.size() * 2
        elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
            if len(edges) != 1:
                print("node %s (%s) has an unexpected number of edges (%d edges)" % (node.name, node.get_type(), len(edges)))
            assert (len(edges) == 1)
            ifm_shape = edges[0].blob.shape
            return (ifm_shape[1] * ifm_shape[2] * ifm_shape[3])
        else:
            return ''

    def get_ofm_size(self, edge):
        ofm_size = 0
        if edge.blob.shape and edge.src_node.role != "Modifier":
            ofm_size = edge.blob.size()
        return ofm_size

    # todo: move this to Topology (per Node class)
    def get_MACs(self, node, ofms_descriptor, tplgy):
        if node.type in ['Convolution', 'Convolution_ReLU']:
            edges = tplgy.find_incoming_edges(node)
            assert (len(edges) == 1)
            num_ifms = edges[0].blob.shape[1]
            if node.type == 'Convolution_ReLU':
                node = node.node1
            return node.get_MACs(ofms_descriptor, num_ifms)
        elif node.type in ['InnerProduct', 'InnerProduct_ReLU']:
            return self.get_weight_size(node, tplgy)
        else:
            return node.get_MACs()#(ofms_descriptor, num_ifms)

    def get_MACs_to_BW(self, node, ofms_descriptor, tplgy):
        pass
        
    def print_inventory(self, tplgy):
        node_types_cnt = tplgy.get_inventory()
        self.file.write('Type, Count\n')
        for type in node_types_cnt:
            line = type + ',' + str(node_types_cnt[type]) + '\n'
            self.file.write(line)
        self.file.write('\n')

    def print_unique(self, unique_layers_list):
        for node in unique_layers_list:
            self.file.write(self.print_node(node[0]) + '\n')

    def print_unique_all(self, unique_layers_dict):
        self.file.write('Type, Configuration\n')
        for type_name in unique_layers_dict:
            self.print_unique(unique_layers_dict[type_name])
        self.file.write('\n')

    def print_bfs(self, tplgy):
        self.file.write(', '.join(self.cols))
        self.file.write('\n')
        tplgy.traverse(None, lambda edge: self.print_edge_cb(edge, tplgy))
        print('cvs printer: done with %s' % self.file.name)

    def get_col_handlers(self, edge, tplgy):
        col_handlers = {
            'Node': (edge.src_node.name if edge.src_node else ''),
            'Type': (str(self.print_node(edge.src_node)) if edge.src_node else ','),
            'Node Details': '#',
            'IFMz': self.print_ifms(edge.src_node, tplgy),
            'IFMy': '#',
            'IFMx': '#',
            'OFMz': (str(edge.blob.shape[1]) if edge.blob.shape else ''),  # OFMz
            'OFMy': (str(edge.blob.shape[2]) if edge.blob.shape else ''),  # OFMy
            'OFMx': (str(edge.blob.shape[3]) if edge.blob.shape else ''),  # OFMx
            'IFM Volume (elems)': str(self.get_ifm_size(edge.src_node, tplgy)),
            'OFM Volume (elems)': str(self.get_ofm_size(edge)),
            'Weights Volume (elems)': str(self.get_weight_size(edge.src_node, tplgy)),
            'Bias Volume (elems)': str(self.get_bias_size(edge.src_node, tplgy)),
            'MACs': str(self.get_MACs(edge.src_node, edge.blob.shape, tplgy)),
        }
        return col_handlers

    def write_to_file(self, col_handlers):
        for col in self.cols:
            if col_handlers[col]!='#':
                self.file.write(col_handlers[col] + ',' )
        self.file.write('\n');

    def print_edge_cb(self, edge, tplgy):
        # If we've printed the contribution of this BLOB, then we skip it.
        # This will naturally filter out ReLU nodes, because they share their
        # BLOB with either Convolution or InnerProduct
        if edge.blob in self.done_blobs:
            #print("skipping BLOB: %s from edge %s" % (edge.blob, str(edge)))
            return
        # We don't want to see 'modifier' nodes (e.g. Concat) it in the CSV, since
        # they contribute no data transfer information
        if edge.src_node.role == 'Modifier':
            return
        self.done_blobs.append(edge.blob)

        col_handlers = self.get_col_handlers(edge, tplgy)
        """
        Add your own handler
        """
        new_col_handlers = col_handlers
        self.write_to_file(new_col_handlers)