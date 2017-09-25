from __future__ import print_function
from .globals import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import topology
from sys import exit
import yaml
"""
pydot is not supported under python 3 and pydot2 doesn't work properly.
pydotplus works nicely (pip install pydotplus)
"""
try:
    # Try to load pydotplus
    import pydotplus as pydot
except ImportError:
    import pydot

def choose_style_by_layertype(layertype, theme):
    try:
        layer_style = theme[layertype]
    except:
        layer_style = theme['layer_default']

    return layer_style


class PngPrinter(object):
    """The printer prints to PNG files"""

    def __init__(self, args, png_prefs, net):
        self.output_image_file = args.infile + '.png'
        self.output_inventory_file = args.infile + '_inventory.png'
        self.caffe_net = net
        self.pydot_nodes = {}
        self.pydot_edges = []
        self.__prefs = png_prefs[png_prefs['preferences']]
        self.__theme = png_prefs[png_prefs['theme']]

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
            'Tensor': PngPrinter.print_tensor,
        }

        printer = layers.get(node.type, PngPrinter.print_default)
        node_label = printer(node, separator, format)
        return node_label

    @staticmethod
    def print_default(node, separator, format):
        node_label = '%s%s(%s)' % (node.name, separator, node.type)
        return node_label

    @staticmethod
    def print_tensor(node, separator, format):
        node_label = '%s%s(%s)' % (node.name, separator, node.type)
        node_label += '\n' + str(node.shape) if node.shape is not None else 'None'
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
            #if node.ceiling:
            #    node_label += ' ceiling'
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
        layer_style = choose_style_by_layertype(node.type, self.__theme)
        if rankdir in ('TB', 'BT'):
            # If graph orientation is vertical, horizontal space is free and
            # vertical space is not; separate words with spaces
            separator = ' '
        else:
            # If graph orientation is horizontal, vertical space is free and
            # horizontal space is not; separate words with newlines
            separator = '\\n'

        node_label = self.get_node_label(node, separator, self.__prefs['node_label'])
        #print('[png_printer] adding node: ', node.name)
        self.pydot_nodes[node.name] = pydot.Node(node_label, **layer_style)

    def add_pydot_edge(self, edge, tplgy):
        # print("adding edge:", edge)
        if (edge.src is None) or (edge.dst is None):
            return

        if False: # self.__prefs['label_edges'] and edge.blob != None:
            edge_label = str(edge.blob.shape)
            # Add parent BLOB name
            if edge.blob.parent is not None:
                edge_label += '\n(' + edge.blob.parent.name + ')'
        else:
            edge_label = '""'

        src_name = edge.src.name
        memory = '' #edge.blob.memory if hasattr(edge.blob, 'memory') else ''

        self.pydot_edges.append({'src': src_name,
                                 'dst': edge.dst.name,
                                 'label': edge_label,
                                 'memory': memory})

    def draw_subgraphs(self, tplgy, pydot_graph):
        nodes = []
        tplgy.traverse(lambda node: nodes.append(node))
        clusters = {}
        for node in nodes:
            pydot_node = self.pydot_nodes[node.name]

            if not hasattr(node, 'subgraph'):
                pydot_graph.add_node(pydot_node)
                continue

            cluster_name = str(node.subgraph)
            if cluster_name in clusters:
                cluster = clusters[cluster_name]
            else:
                # New subgraph
                cluster = pydot.Cluster(cluster_name, label=cluster_name, style="filled")
                clusters[cluster_name] = cluster
            cluster.add_node(pydot_node)

        for _,cluster in clusters.items():
            pydot_graph.add_subgraph(cluster)
        pydot_graph.write_raw('debug.dot')

    def draw_net(self, caffe_net, rankdir, tplgy):
        pydot_graph = pydot.Dot(self.caffe_net.name if self.caffe_net.name else 'Net',
                                graph_type='digraph',
                                compound='true',
                                rankdir=rankdir)

        tplgy.traverse(lambda node: self.add_pydot_node(node, tplgy, rankdir),
                       lambda edge: self.add_pydot_edge(edge, tplgy))
        self.draw_subgraphs(tplgy, pydot_graph)

        for edge in self.pydot_edges:
            if edge['dst'] not in self.pydot_nodes:
                print('Fatal error: node \'%s\' of edge %s is not in the pydot_node list!' % (edge['dst'], edge))
                #exit()
                break

            style = 'dashed' if edge['memory']=='scratchpad' else ''

            pydot_graph.add_edge(
                pydot.Edge(self.pydot_nodes[edge['src']],
                           self.pydot_nodes[edge['dst']],
                           label=edge['label'],
                           style = style))

        print("Number of nodes:", len(self.pydot_nodes))
        if self.__prefs['gen_dot_file']:
            print('Generating dot file')
            pydot_graph.write_raw('debug.dot')
        return pydot_graph.create_png()

    def print_bfs(self, tplgy):
        with open(self.output_image_file, 'wb') as fid:
            fid.write(self.draw_net(self.caffe_net, self.__prefs['rankdir'], tplgy))
        print('Drawing net to %s' % self.output_image_file)

    def print_inventory(self, inventory):
        self.draw_inventory(inventory)

    def draw_inventory(self, inventory):
        labels = []
        values = []
        for type, count in inventory.items():
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
