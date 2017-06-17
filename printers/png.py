from __future__ import print_function
from .globals import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import topology
from sys import exit

"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

# options
default_options = {
    'verbose': True,
    # The node label refers to the text that is inside each node in the graph
    'node_label': 'custom', # {'custom', 'caffe', 'minimal'}
    # Annotate the edges with the BLOB sizes
    'label_edges': True,
    # Graph drawing direction: left-right, top-bottom, bottom-top
    'rankdir': 'TB',  # {'LR', 'TB', 'BT'}
    # Draw cluster bounding boxes
    'draw_clusters': True,
    # The separator character for parsing a node name to cluster_name-separator-node
    # (only relevant if draw_clusters is True)
    'cluster_name_separator': '/',
    # Generate a dot file (useful for debugging dot generation errors
    'gen_dot_file': True,
}

google_options = {
    'verbose': True,
    # The node label refers to the text that is inside each node in the graph
    'node_label': 'custom', # {'custom', 'caffe', 'minimal'}
    # Annotate the edges with the BLOB sizes
    'label_edges': True,
    # Graph drawing direction: left-right, top-bottom, bottom-top
    'rankdir': 'TB',  # {'LR', 'TB', 'BT'}
    # Draw cluster bounding boxes
    'draw_clusters': False,
    # The separator character for parsing a node name to cluster_name-separator-node
    # (only relevant if draw_clusters is True)
    'cluster_name_separator': '/',
    # Generate a dot file (useful for debugging dot generation errors
    'gen_dot_file': True,
}

# Themes
CAFFE_THEME = {
    'layer_default': {'shape': 'record',
                      'fillcolor': '#6495ED',
                      'style': 'filled'},

    'Convolution':  {'shape': 'record',
                     'fillcolor': '#FF5050',
                     'style': 'filled'},

    'Pooling':      {'shape': 'record',
                     'fillcolor': '#FF9900',
                     'style': 'filled'},

    'InnerProduct': {'shape': 'record',
                     'fillcolor': '#CC33FF',
                     'style': 'filled'},
}

SOFT_THEME = {
    'layer_default': {'shape': 'record',
                      'fillcolor': '#6495ED',
                      'style': 'rounded, filled'},

    'Convolution':   {'shape': 'record',
                      'fillcolor': '#FF5050',
                      'style': 'rounded, filled'},

    'Pooling':       {'shape': 'record',
                      'fillcolor': '#FF9900',
                      'style': 'rounded, filled'},

    'InnerProduct':  {'shape': 'record',
                      'fillcolor': '#CC33FF',
                      'style': 'rounded, filled'},

    'InnerProduct_ReLU': {'shape': 'record',
                     'fillcolor': '#CC33FF',
                     'style': 'rounded, filled'},

    'Concat':        {'shape': 'box3d',
                      'fillcolor': 'gray',
                      'style': 'filled'},

    'Softmax':        {'shape': 'record',
                      'fillcolor': 'yellow',
                      'style': 'rounded, filled'},

    'Convolution_ReLU':{'shape': 'record',
                      'fillcolor': 'coral3',
                      'style': 'rounded, filled'},

    'Convolution_ReLU_Pooling':
                      {'shape': 'record',
                       'fillcolor': 'darkslategray',
                       'style': 'rounded, filled'},

}

# http://designpieces.com/palette/google-new-logo-2015-color-palette-hex-and-rgb/
GOOGLE_THEME = {
    'layer_default': {'shape': 'record',
                      'fillcolor': '#3cba54',
                      'style': 'filled'},

    'Convolution':   {'shape': 'record',
                      'fillcolor': '#4885ed',
                      'style': 'filled'},

    'Pooling':       {'shape': 'record',
                      'fillcolor': '#db3236',
                      'style': 'filled'},

    'InnerProduct': dict(shape='record', fillcolor='#4885ed', style='filled'),
    'InnerProduct_ReLU': dict(shape='record', fillcolor='#4885ed', style='filled'),

    'Softmax':        {'shape': 'record',
                      'fillcolor': '#f4c20d',
                      'style': 'filled'},

    'SoftmaxWithLoss': dict(shape='record', fillcolor='#f4c20d', style='filled'),

    'Convolution_ReLU':{'shape': 'record',
                      'fillcolor': '#4885ed',
                      'style': 'filled'},

    'Convolution_ReLU_Pooling':
                      {'shape': 'record',
                       'fillcolor': '#4885ed',
                       'style': 'filled'},
    'Input': dict(shape='octagon', fillcolor='lightgray', style='filled'),

}

# theme = CAFFE_THEME
theme = SOFT_THEME
#theme = GOOGLE_THEME
options = google_options
#options = default_options

def choose_style_by_layertype(layertype):
    try:
        layer_style = theme[layertype]
    except:
        layer_style = theme['layer_default']

    return layer_style


class PngPrinter(object):
    """The printer prints to PNG files"""

    def __init__(self, args, net):
        self.output_image_file = args.infile + '.png'
        self.output_inventory_file = args.infile + '_inventory.png'
        self.caffe_net = net
        self.pydot_nodes = {}
        self.pydot_edges = []

    @staticmethod
    def get_node_label(node, separator, format):
        """Define node label based on layer type.
        """

        if format == "minimal":
            return node.name

        layers = {
            'Convolution': PngPrinter.print_conv,
            'Deconvolution': PngPrinter.print_conv,
            'Pooling': PngPrinter.print_pool,
            'LRN': PngPrinter.print_lrn,
            'Reshape': PngPrinter.print_reshape,
            'Eltwise': PngPrinter.print_eltwise,
            'Concat': PngPrinter.print_concat,
            'Convolution_ReLU': PngPrinter.print_mergednode,
            'Convolution_ReLU_Pooling': PngPrinter.print_mergednode,
        }

        printer = layers.get(node.type, PngPrinter.print_default)
        node_label = printer(node, separator, options['node_label'])
        return node_label

    @staticmethod
    def print_default(node, separator, format):
        node_label = '%s%s(%s)' % (node.name, separator, node.type)
        return node_label

    @staticmethod
    def print_concat(node, separator, format):
        node_label = '%s%s(%s)' % (node.name, separator, node.type)
        #node_label = '%s' % (node.type)
        return node_label

    @staticmethod
    def print_mergednode(node, separator, format):
        node1 = PngPrinter.get_node_label(node.node1, separator, format)
        node2 = PngPrinter.get_node_label(node.node2, separator, format)

        node1_list = node1.split(separator)
        node2_list = node2.split(separator)
        #node_label = '%s%s%s' % (node1_list[0], separator, node2_list[0])
        node_label = node1_list[0]          # first node's name
        for desc in node1_list[1:]:
            node_label += separator + desc
        for desc in node2_list:
            node_label += separator + desc
        return node_label

    @staticmethod
    def print_conv(node, separator, format):
        if format == 'caffe':
            # Outer double quotes needed or else colon characters don't parse
            # properly
            node_label = '%s%s(%s)%skernel size: %d%sstride: %d%spad: %d' %\
                         (node.name, separator,
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + 'k=' + str(node.kernel_size) + 'x' + str(
                node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_pool(node, separator, format):
        pooling_type = get_pooling_types_dict()
        if format=='caffe':
            node_label = '%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d' % \
                         (node.name, separator, pooling_type[node.pool_type],
                          node.type, separator,
                          node.kernel_size, separator,
                          node.stride, separator, node.pad)
        elif format == 'custom':
            node_label = node.name + separator + pooling_type[node.pool_type] + ', k=' + str(node.kernel_size) + 'x' + str(
                         node.kernel_size) + '/s=' + str(node.stride) + ' pad=' + str(node.pad)
            if node.ceiling:
                node_label += ' ceiling'
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_lrn(node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = node.name + separator + lrn_type[node.norm_region] + \
                          ' size=' + str(node.local_size) + ' alpha=' + str(
                          node.alpha) + ' beta=' + str(node.beta)
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_reshape(node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = '%s%s[%s,%s,%s,%s]' % \
                         (node.name, separator, node.reshape_param.dim[0], node.reshape_param.dim[1], node.reshape_param.dim[2], node.reshape_param.dim[3])
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    @staticmethod
    def print_eltwise(node, separator, format):
        if format == 'caffe':
            node_label = node.name
        elif format == 'custom':
            node_label = '%s%s%s' % \
                         (node.name, separator, get_eltwise_op_dict()[node.operation])
            node_label = '%s%s(%s)' % (node_label, separator, node.type)
        else:
            node_label = None
        return node_label

    def add_pydot_node(self, node, tplgy, rankdir):
        layer_style = choose_style_by_layertype(node.type)
        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        node_label = self.get_node_label(node, separator, options['node_label'])
        #print('[png_printer] adding node: ', node.name)
        self.pydot_nodes[node.name] = pydot.Node(node_label, **layer_style)

    def add_pydot_edge(self, edge, tplgy):
        if (edge.src_node is None) or (edge.dst_node is None):
            return

        if options['label_edges'] and edge.blob != None:
            edge_label = str(edge.blob.shape) #get_edge_label(edge.src_node)
        else:
            edge_label = '""'

        src_name = edge.src_node.name
        self.pydot_edges.append({'src': src_name,
                                'dst': edge.dst_node.name,
                                'label': edge_label})

    def draw_clusters(self, pydot_graph):
        clusters = {}
        for node_name, pydot_node in self.pydot_nodes.iteritems():
            separator = options['cluster_name_separator']
            if node_name.find(separator) > 0:
                cluster_name = node_name[0:node_name.find(separator)]
            else:
                cluster_name = node_name
            # Dot doesn't handle well clusters with names that contain '-'
            cluster_name = cluster_name.replace('-', '_')

            is_new_cluster = True
            for name, cluster in clusters.iteritems():
                if name == cluster_name:
                    is_new_cluster = False
                    cluster.add_node(pydot_node)
                    break
            if is_new_cluster:
                cluster = pydot.Cluster(cluster_name, label=cluster_name)
                clusters[cluster_name] = cluster
                # print("creating cluster: ", cluster_name)
                cluster.add_node(pydot_node)

        for cluster in clusters.itervalues():
            # for clusters of size 1, we don't want to draw a cluster box
            if len(cluster.get_nodes()) == 1:
                pydot_graph.add_node(cluster.get_nodes()[0])
            else:
                pydot_graph.add_subgraph(cluster)

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                compound='true',
                                rankdir=rankdir)

        # tplgy.dump_edges()
        tplgy.traverse(lambda node: self.add_pydot_node(node, tplgy, rankdir),
                       lambda edge: self.add_pydot_edge(edge, tplgy))

        # Cluster nodes by name prefix
        # add the nodes and edges to the graph.
        if options['draw_clusters']:
            self.draw_clusters(pydot_graph)
        else:   # not clustering
            for pydot_node in iter(self.pydot_nodes.values()):
                pydot_graph.add_node(pydot_node)

        for edge in self.pydot_edges:
            if edge['dst'] not in self.pydot_nodes:
                print('Fatal error: node \'%s\' of edge %s is not in the pydot_node list!' % (edge['dst'], edge))
                #exit()
                break
            pydot_graph.add_edge(
                pydot.Edge(self.pydot_nodes[edge['src']],
                           self.pydot_nodes[edge['dst']],
                           label=edge['label']))

        print("Number of nodes:", len(self.pydot_nodes))
        if options['gen_dot_file']:
            print('Generating dot file')
            pydot_graph.write_raw('debug.dot')
        return pydot_graph.create_png()

    def print_bfs(self, tplgy):
        with open(self.output_image_file, 'wb') as fid:
            fid.write(self.draw_net(self.caffe_net, options['rankdir'], tplgy))
        print('Drawing net to %s' % self.output_image_file)

    def print_inventory(self, tplgy):
        self.draw_inventory(tplgy)

    def draw_inventory(self, tplgy):
        node_types_cnt = tplgy.get_inventory()

        labels = []
        values = []
        for type, count in node_types_cnt.iteritems():
            labels.append(type)
            values.append(count)
        data = values

        xlocations = np.array(range(len(data))) + 0.5
        width = 0.5
        plt.bar(xlocations, data, width=width)
        #plt.yticks(range(0, max(values)))
        plt.yticks(np.arange(min(values), max(values), (max(values) - min(values)) / 10 ))
        plt.xticks(xlocations + width / 2, labels, rotation='vertical')
        # Pad margins so that markers don't get clipped by the axes
        #plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.25)

        plt.xlim(0, xlocations[-1] + width * 2)
        plt.title("Nodes Inventory")
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()

        plt.savefig(self.output_inventory_file)
        print('Drawing net to %s' % self.output_inventory_file)
