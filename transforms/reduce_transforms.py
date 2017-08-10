"""This module contains reduction transforms, which reduce the
graph to a summarized format of some sort.
For example, the Inventory transformer, reduces the graph
to a nodes type histogram.
"""
from collections import Counter

def __is_unique(node, unique_list):
    unique = True
    for unique_layer in unique_list:
        if unique_layer[0].is_same(node):
            unique = False
            unique_list.remove(unique_layer)
            unique_list.append((node, unique_layer[1]+1))
            break
    return unique


def __add_unique(node, unique_layers):
    if unique_layers.get(node.type)==None:
        unique_layers[node.type] = []
    if __is_unique(node, unique_layers[node.type]):
        unique_layers[node.type].append((node, 1))

def get_uniques_inventory(tplgy):
    ''' This transform creates a dictionary:
    unique[node_type] <== #instances
    Use this to get a distribution function (histogram)
    of unique node types in the graph.
    Node types are more detailed than in a plain inventory reduction.
    '''
    unique_nodes = {}
    tplgy.traverse(lambda node: __add_unique(node, unique_nodes))
    return unique_nodes

def get_inventory(tplgy):
    ''' This transform creates a dictionary:
    inventory[node_type] <== #instances
    Use this to get a distribution function (histogram)
    of each node type in the graph.
    '''
    node_cnt = []
    tplgy.traverse(lambda node: node_cnt.append(node.type))
    return Counter(node_cnt)
