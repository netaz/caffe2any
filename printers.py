"""
Classes that print to different media
"""

# See - http://stackoverflow.com/questions/2970858/why-doesnt-print-work-in-a-lambda
from __future__ import print_function
from collections import Counter
import caffe_pb2 as caffe

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

pooling_type = {0: 'MAX', 1: 'AVG', 2: 'STOCHASTIC'}
lrn_type = {0: 'ACROSS_CHANNELS', 1: 'WITHIN_CHANNEL'}
PIXEL_SIZE_BYTES = 2  # TODO: make this configrable


class ConsolePrinter:
    """A simple console printer"""

    def __init__(self):
        pass

    def print_pool(self, node):
        desc = pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
            node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
        if node.ceiling:
            desc += ' ceiling'
        return 'Pooling', desc

    def print_deconv(self, node):
        return 'Deconvolution', \
               'k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(
                   node.pad)

    def print_conv(self, node):
        return 'Convolution', \
               'k=' + str(node.kernel_size) + "x" + str(node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(
                   node.pad)

    def print_lrn(self, node):
        return 'LRN', \
               lrn_type[node.norm_region] + ' size=' + str(node.local_size) + ' alpha=' + str(
                   node.alpha) + ' beta=' + str(node.beta)

    def print_unknown(self, node):
        return node.type, ""

    def print_layer(self, node, count):
        print_fn = {
            "Pooling": self.print_pool,
            "Convolution": self.print_conv,
            "Deconvolution": self.print_deconv,
            "LRN": self.print_lrn,
        }.get(node.type, self.print_unknown)
        row_format = "{:<20} {:<45} {:<40}"  # 3 is the number of cols
        # print('\t%-20s%-3s Count=%-10d' % print_fn(layer, count))
        print(row_format.format(*(print_fn(node) + (count,))))

    def print_inventory(self, tplgy):
        print("Inventory:\n----------")
        node_types_cnt = tplgy.nodes_count()
        for type in node_types_cnt:
            print('\t%-20s%-3i' % (type, node_types_cnt[type]))
        print("Total=", len(tplgy.nodes))
        print("")

    def print_unique(self, unique_layers_list):
        for node in unique_layers_list:
            self.print_layer(node[0], node[1])  # print node, count

    def print_unique_all(self, unique_layers_dict):
        print("Unique:\n--------")
        for type_name in unique_layers_dict:
            self.print_unique(unique_layers_dict[type_name])
        print("")

    def print_bfs(self, tplgy):
        tplgy.traverse(lambda node: print(str(node)),
                       lambda edge: print('\t' + str(edge)))


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

    def print_MACs(self, node, ofms_descriptor, tplgy):
        if node.type in ['Convolution']:
            edges = tplgy.find_incoming_edges(node)
            assert (len(edges) == 1)

            num_ifms = edges[0].blob.shape[1]

            # macs = #OFMs*OFM_X*OFM_Y*#IFMs*K_X*K_Y
            num_ofms = ofms_descriptor[1]
            ofm_x = ofms_descriptor[2]
            ofm_y = ofms_descriptor[3]
            MACs = num_ofms * ofm_x * ofm_y * num_ifms * node.kernel_size * node.kernel_size
            return str(MACs)
        elif node.type in ['InnerProduct']:
            return str(self.get_weight_size(node, tplgy))
        else:
            return ''

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
            self.print_MACs(edge.src_node, edge.blob.shape, tplgy) + ',' +  # MACs
            '\n')


# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'box3d',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}


def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d

# optional - caffe color scheme
def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type.
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


class PngPrinter:
    """The printer prints to PNG files"""

    def __init__(self, args, net):
        self.output_image_file = filename = args.infile + '.png'
        self.caffe_net = net
        self.pydot_nodes = {}
        self.pydot_edges = []

        print('Drawing net to %s' % self.output_image_file)

    def get_layer_label(self, layer, rankdir, verbose):
        """Define node label based on layer type.

        Parameters
        ----------
        layer : ?
        rankdir : {'LR', 'TB', 'BT'}
            Direction of graph layout.

        Returns
        -------
        string :
            A label for the current layer
        """

        if verbose==False:
            return layer.name

        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        if layer.type == 'Convolution' or layer.type == 'Deconvolution':
            # Outer double quotes needed or else colon characters don't parse
            # properly
            node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                         (layer.name,
                          separator,
                          layer.type,
                          separator,
                          layer.kernel_size,
                          separator,
                          layer.stride,
                          separator,
                          layer.pad)
        elif layer.type == 'Pooling':
            #pooling_types_dict = get_pooling_types_dict()
            node_label = self.print_pool(layer, separator)
            """
            node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                         (layer.name,
                          separator,
                          pooling_types_dict[layer.pool_type],
                          layer.type,
                          separator,
                          layer.kernel_size,
                          separator,
                          layer.stride,
                          separator,
                          layer.pad)
                          """
        else:
            node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
        return node_label


    def print_pool(self, node, separator):
        #optional: compact/verbose
#        desc = pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
#            node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
#        if node.ceiling:
#            desc += ' ceiling'
#        return '"%s%s(%s)"' % (desc, separator, 'Pooling')
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' % \
                     (node.name,
                      separator,
                      pooling_types_dict[node.pool_type],
                      node.type,
                      separator,
                      node.kernel_size,
                      separator,
                      node.stride,
                      separator,
                      node.pad)
        return node_label

    def merge_nodes(self, src_node, dst_node):
        self.nodes_to_merge[src_node] = {dst_node}

    def add_pydot_node(self, node, tplgy, rankdir):
        #node_name = "%s_%s" % (node.name, node.type)
        #self.pydot_nodes[node.name] = pydot.Node(node.name,
                                    #    **NEURON_LAYER_STYLE)
        layer_style = LAYER_STYLE_DEFAULT
        layer_style['fillcolor'] = choose_color_by_layertype(node.type)
        # optional: verbosity
        node_label = self.get_layer_label(node, rankdir, verbose=True)
        self.pydot_nodes[node.name] = pydot.Node(node_label, **layer_style)

    def add_pydot_edge(self, edge, tplgy):
        if (edge.src_node is None) or (edge.dst_node is None):
            return
        #optional
        label_edges = True
        if label_edges and edge.blob != None:
            edge_label = str(edge.blob.shape) #get_edge_label(edge.src_node)
        else:
            edge_label = '""'

        src_name = edge.src_node.name
        self.pydot_edges.append({'src': src_name,
                                'dst': edge.dst_node.name,
                                'label': edge_label})

    def filter_relu_node(self, node, tplgy):
        if node.type == "ReLU":
            incoming_edges = tplgy.find_incoming_edges(node)
            outgoing_edges = tplgy.find_outgoing_edges(node)
            for incoming_edge in incoming_edges:
                src = incoming_edge.src_node
                src.name += "  +  " + node.name + "\n"
                for outgoing_edge in outgoing_edges:
                    tplgy.add_edge(src, outgoing_edge.dst_node, incoming_edge.blob)
                    #print("adding " + str(src) + "->" + str(outgoing_edge.dst_node))
                tplgy.del_edge(incoming_edge)

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                rankdir=rankdir)

        # optional: collapse ReLU nodes
        collapse_relu = True
        if collapse_relu:
            tplgy.traverse(lambda node: self.filter_relu_node(node, tplgy))

        tplgy.traverse(lambda node: self.add_pydot_node(node, tplgy, rankdir),
                       lambda edge: self.add_pydot_edge(edge, tplgy))


        # add the nodes and edges to the graph.
        for node in self.pydot_nodes.values():
            pydot_graph.add_node(node)

        for edge in self.pydot_edges:
            pydot_graph.add_edge(
                pydot.Edge(self.pydot_nodes[edge['src']],
                           self.pydot_nodes[edge['dst']],
                           label=edge['label']))

        print("number of nodes:", len(self.pydot_nodes))
        return pydot_graph.create_png()

    def print_bfs(self, tplgy):
        self.done_blobs = []

        rankdir = 'TB'  # {'LR', 'TB', 'BT'}

        with open(self.output_image_file, 'wb') as fid:
            fid.write(self.draw_net(self.caffe_net, rankdir, tplgy))
