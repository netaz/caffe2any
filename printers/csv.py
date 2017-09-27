import topology
from .globals import *
pooling_type = get_pooling_types_dict()
from transforms import decorator_transforms

class CsvPrinter:
    """A CSV file printer"""

    def __init__(self, outfile):
        self.file = open(outfile + '.csv', "wt")
        self.done_nodes = []
        # TODO - READ THIS FROM CONFIGURATION
        self.cols = ['Node', 'Type', 'Node Details', 'IFMz', 'IFMy', 'IFMx', 'OFMz', 'OFMy', 'OFMx',
                     'IFM Volume (elems)', 'OFM Volume (elems)', 'Weights Volume (elems)', 'Bias Volume (elems)', 'BW', 'MACs', 'MACs/element']

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

    def print_eltwise_relu(selfself, merged_node):
        node = merged_node.node1
        op_lookup = get_eltwise_op_dict()
        return 'Eltwise/ReLU,' + op_lookup[node.operation]

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
            "Eltwise_ReLU": self.print_eltwise_relu,
        }.get(node.type, self.print_unknown)
        return print_fn(node)

    def print_ifms(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Convolution_ReLU', 'InnerProduct', 'InnerProduct_ReLU',
                         'Pooling', 'Deconvolution', 'LRN', 'Softmax']:
            assert len(edges)==1
            assert type(edges[0].src)==topology.BLOB
            ifm_shape = edges[0].src.shape
            if ifm_shape is None:
                #print("node " + node.name + " has no ifm_shape")
                return ',,'
            return str(ifm_shape[1]) + ',' + str(ifm_shape[2]) + ',' + str(ifm_shape[3])
        elif node.type in ['Eltwise', 'Eltwise_ReLU']:
            assert len(edges)==2
            for edge in edges:
                assert type(edge.src)==topology.BLOB
            ifm_shape = edges[0].src.shape
            return str(ifm_shape[1]) + ',' + str(ifm_shape[2]) + ',' + str(ifm_shape[3])
        else:
            return ',,'

    def print_ofms(self, node, tplgy):
        edges = tplgy.find_outgoing_edges(node)
        if node.type in ['Convolution', 'Convolution_ReLU', 'InnerProduct', 'InnerProduct_ReLU',
                         'Pooling', 'Deconvolution', 'Eltwise', 'Eltwise_ReLU', 'LRN', 'Softmax']:
            assert len(edges)==1
            assert type(edges[0].dst)==topology.BLOB
            ofm_shape = edges[0].dst.shape
            if ofm_shape is None:
                return ',,'
            return str(ofm_shape[1]) + ',' + str(ofm_shape[2]) + ',' + str(ofm_shape[3])
        else:
            return ',,'

    def get_MACs_to_BW(self, node, ofms_descriptor, tplgy):
        pass

    def print_inventory(self, inventory):
        self.file.write('Type, Count\n')
        for type in inventory:
            line = type + ',' + str(inventory[type]) + '\n'
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
        decorator_transforms.add_size_annotations(tplgy)
        decorator_transforms.add_macs_annotations(tplgy)

        self.file.write(', '.join(self.cols))
        self.file.write('\n')
        tplgy.traverse(lambda node: self.print_node_cb(node, tplgy))
        print('cvs printer: done with %s' % self.file.name)

    def get_col_handlers(self, node, tplgy):
        col_handlers = {
            'Node': (node.name if node else ''),
            'Type': (str(self.print_node(node)) if node else ','),
            'Node Details': '#',
            'IFMz': self.print_ifms(node, tplgy),
            'IFMy': '#',
            'IFMx': '#',
            'OFMz': self.print_ofms(node, tplgy),
            'OFMy': '#',
            'OFMx': '#',
            #'OFMz': (str(edge.blob.shape[1]) if edge.blob.shape else ''),  # OFMz
            #'OFMy': (str(edge.blob.shape[2]) if edge.blob.shape else ''),  # OFMy
            #'OFMx': (str(edge.blob.shape[3]) if edge.blob.shape else ''),  # OFMx
            'IFM Volume (elems)':str(node.get_attr('ifm_size')),
            'OFM Volume (elems)': str(node.get_attr('ofm_size')),
            'Weights Volume (elems)': str(node.get_attr('weights_size')),
            'Bias Volume (elems)': str(node.get_attr('bias_size')),
            'BW': str(node.get_attr('bw')),
            'MACs': str(node.get_attr('macs')),
            'MACs/element': str(node.get_attr('macs/bw'))
        }
        return col_handlers

    def write_to_file(self, col_handlers):
        for col in self.cols:
            if col_handlers[col]!='#':
                self.file.write(col_handlers[col] + ',' )
        self.file.write('\n');

    def print_node_cb(self, node, tplgy):
        # If we've printed the contribution of this BLOB, then we skip it.
        # This will naturally filter out ReLU nodes, because they share their
        # BLOB with either Convolution or InnerProduct
        if (node in self.done_nodes) or type(node)==topology.BLOB:
            #print("skipping BLOB: %s from edge %s" % (edge.blob, str(edge)))
            return
        # We don't want to see 'modifier' nodes (e.g. Concat) it in the CSV, since
        # they contribute no data transfer information
        #if edge.src_node.role == 'Modifier':
        #    return
        self.done_nodes.append(node)

        col_handlers = self.get_col_handlers(node, tplgy)
        """
        Add your own handler
        """
        new_col_handlers = col_handlers
        self.write_to_file(new_col_handlers)
