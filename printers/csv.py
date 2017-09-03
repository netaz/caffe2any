from .globals import *
pooling_type = get_pooling_types_dict()
from transforms import decorator_transforms

class CsvPrinter:
    """A CSV file printer"""

    def __init__(self, fname):
        self.file = open(fname, "wt")
        self.done_blobs = []
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
            'IFM Volume (elems)':str(edge.src_node.get_attr('ifm_size')),
            'OFM Volume (elems)': str(edge.src_node.get_attr('ofm_size')),
            'Weights Volume (elems)': str(edge.src_node.get_attr('weights_size')),
            'Bias Volume (elems)': str(edge.src_node.get_attr('bias_size')),
            'BW': str(edge.src_node.get_attr('bw')),
            'MACs': str(edge.src_node.get_attr('macs')),
            'MACs/element': str(edge.src_node.get_attr('macs/bw'))
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
