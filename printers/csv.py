from globals import *
pooling_type = get_pooling_types_dict()

class CsvPrinter:
    """A CSV file printer"""

    def __init__(self, fname):
        self.file = open(fname, "wt")

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

    def print_lrn(self, node):
        return 'LRN,' + lrn_type[node.norm_region] + ' local_size=' + str(node.local_size) + ' alpha=' + str(
            node.alpha) + ' beta=' + str(node.beta)

    def print_unknown(self, node):
        return str(node.type) + ','

    def print_layer(self, node):
        print_fn = {
            "Pooling": self.print_pool,
            "Convolution": self.print_conv,
            "Deconvolution": self.print_deconv,
            "LRN": self.print_lrn,
        }.get(node.type, self.print_unknown)
        return print_fn(node)

    def print_ifms(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Pooling']:
            assert (len(edges) == 1)
            ifm_shape = edges[0].blob.shape
            return str(ifm_shape[1]) + ',' + str(ifm_shape[2]) + ',' + str(ifm_shape[3])
        elif node.type in ['InnerProduct']:
            assert (len(edges) == 1)
            ifm_shape = edges[0].blob.shape
            return str(ifm_shape[1]) + ',' + str(ifm_shape[2]) + ',' + str(ifm_shape[3])
        else:
            return ',,'

    def get_weight_size(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution']:
            assert (len(edges) == 1)
            num_ifms = edges[0].blob.shape[1]
            return node.kernel_size * node.kernel_size * node.num_output * num_ifms
        elif node.type in ['InnerProduct']:
            assert (len(edges) == 1)
            return (self.get_ifm_size(node, tplgy) * node.num_output)
        else:
            return 0

    def get_bias_size(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'InnerProduct']:
            return node.num_output
        else:
            return 0

    def get_ifm_size(self, node, tplgy):
        edges = tplgy.find_incoming_edges(node)
        if node.type in ['Convolution', 'Pooling']:
            assert (len(edges) == 1)
            ifm = edges[0].blob
            return ifm.size()
        elif node.type in ['InnerProduct']:
            assert (len(edges) == 1)
            ifm_shape = edges[0].blob.shape
            return (ifm_shape[1] * ifm_shape[2] * ifm_shape[3])
        else:
            return ''

    def get_MACs(self, node, ofms_descriptor, tplgy):
        if node.type in ['Convolution']:
            edges = tplgy.find_incoming_edges(node)
            assert (len(edges) == 1)
            num_ifms = edges[0].blob.shape[1]
            return node.get_MACs(ofms_descriptor, num_ifms)
        elif node.type in ['InnerProduct']:
            return self.get_weight_size(node, tplgy)
        else:
            return node.get_MACs()

    def get_MACs_to_BW(self, node, ofms_descriptor, tplgy):
        pass
        
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
        self.file.write(
            'Node, Type, Node Details,IFMz,IFMy,IFMx,OFMz,OFMy,OFMx, IFM Size (pixels), OFM Size (pixels), Weights Size(pixels), Bias Size(pixels), MACs\n')
        self.done_blobs = []
        tplgy.traverse(None, lambda edge: self.print_edge_cb(edge, tplgy))

    def print_edge_cb(self, edge, tplgy):
        if edge.blob in self.done_blobs:
            return  # been there, done that

        self.done_blobs.append(edge.blob)
        ofm_size = 0
        if edge.blob.shape and edge.src_node.role != "Modifier":
            ofm_size = edge.blob.size()

        self.file.write(
            (edge.src_node.name if edge.src_node else '') + ',' +  # Node name
            (str(self.print_layer(edge.src_node)) if edge.src_node else ',') + ',' +  # Layer type, details
            self.print_ifms(edge.src_node, tplgy) + ',' +  # IFM
            (str(edge.blob.shape[1]) if edge.blob.shape else '') + ',' +  # OFMz
            (str(edge.blob.shape[2]) if edge.blob.shape else '') + ',' +  # OFMy
            (str(edge.blob.shape[3]) if edge.blob.shape else '') + ',' +  # OFMx
            str(self.get_ifm_size(edge.src_node, tplgy)) + ',' +  # IFM size - pixels
            str(ofm_size) + ',' +  # OFM size - pixels
            str(self.get_weight_size(edge.src_node, tplgy)) + ',' +  # Weights size - pixels
            str(self.get_bias_size(edge.src_node, tplgy)) + ',' +  # Bias size - pixels
            str(self.get_MACs(edge.src_node, edge.blob.shape, tplgy)) + ',' +  # MACs
            #str(self.get_MACs_to_BW(edge.src_node, edge.blob.shape, tplgy)) + ',' +  # MACs to BW ratio
            '\n')