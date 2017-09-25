#!/usr/bin/python

"""

"""
import sys
import argparse
from collections import deque, Counter
import caffe_pb2 as caffe
from google.protobuf import text_format
from printers import csv, console, png
from parsers.caffe_parser import parse_caffe_net
from parsers.caffe2_parser import parse_caffe2_net
from transforms import reduce_transforms
import topology
import yaml
import logging

''' This is a crude dynamic load of printer classes.
In the future, need to make this nicer.
This provides the ability to dynamically load printers from other
code bases.
'''
import inspect
import importlib

def load_printer(printer_type, my_class=None):
    if my_class == None:
        module = importlib.import_module('printers')
        return getattr(module, printer_type)

    mod_name = 'printers.{0}'.format(printer_type)
    try:
        module = importlib.import_module(mod_name)
        return getattr(module, my_class)
    except ImportError:
        return None


def sum_blob_mem(tplgy, node, blobs, sum):
    if node.type == "Input" or node.role == "Modifier":
        return
    out_edges = tplgy.find_outgoing_edges(node)
    for out_edge in out_edges:
        if out_edge.blob not in blobs:
            shape = out_edge.blob.shape
            sum[0] += out_edge.blob.size()
            blobs.append(out_edge.blob)

from transforms.update_blobs_sizes import update_blobs_sizes
from transforms import fold_transforms
from transforms import decorator_transforms

def __prune_edges(tplgy):
    ''' Remove unnecessary edges '''
    pruned_edges = []
    tplgy.traverse(None, lambda edge: pruned_edges.append(edge if type(edge.src)==type(edge.dst) else None))
    for edge in pruned_edges:
        if edge is not None:
            tplgy.del_edge(edge)

def apply_transforms(prefs, tplgy):
    ''' Handle optional transform processing on the topology
    '''
    if prefs['remove_dropout']:
        tplgy.remove_op_by_type('Dropout')
        __prune_edges(tplgy)
    if prefs['merge_conv_relu']:
        tplgy.merge_ops('Convolution', 'ReLU')

    if prefs['merge_ip_relu']:
        tplgy.merge_ops('InnerProduct', 'ReLU')
    if prefs['merge_conv_relu_pooling']:
        tplgy.merge_ops('Convolution_ReLU', 'Pooling')

    '''
    tplgy.dump_edges()
    '''
    fold_transforms.concat_removal(tplgy)
    #__prune_edges(tplgy)
    return

    if prefs['fold_batchnorm']:
        fold_transforms.fold_pair(tplgy, 'Convolution', 'BatchNorm')
    if prefs['fold_scale']:
        fold_transforms.fold_pair(tplgy, 'Convolution', 'Scale')

    if prefs['merge_sum_relu']:
        tplgy.merge_nodes('Eltwise', 'ReLU')
    #decorator_transforms.horizontal_fusion(tplgy)


import logging.config
logging.config.fileConfig('logging.conf')

def main():
    print("caffe2any v0.5")
    logger = logging.getLogger('topology')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--printer', help='output printer (csv, console, png)', default='console')
    parser.add_argument('-d', '--display', type=str, help='display inventory,unique,output,bfs,mem')
    parser.add_argument('infile', help='input prototxt file')
    args = parser.parse_args()

    net = caffe.NetParameter()

    # Read a Caffe prototxt file
    try:
        f = open(sys.argv[1], "rb")
        text_format.Parse(f.read(), net)
        f.close()
    except IOError:
        exit("Could not open file " + sys.argv[1])

    tplgy = parse_caffe_net(net)
    #.dump_blobs()
    #exit()
    # calculate BLOBs sizes
    #tplgy.traverse(lambda node: update_blobs_sizes(tplgy, node))
    #tplgy.dump_edges()
    #exit()

    # read preferences
    with open("caffe2any_cfg.yml", 'r') as cfg_file:
        prefs = yaml.load(cfg_file)

    apply_transforms(prefs['transforms'], tplgy)

    for printer_str in args.printer.split(','):
        if printer_str == 'console':
            printer = console.ConsolePrinter()
        elif printer_str == 'png':
            printer = png.PngPrinter(args, prefs['png'], net)
        elif printer_str == 'csv':
            printer = csv.CsvPrinter(args.infile + '.csv')
        else:
            printer_ctor = load_printer(printer_str, 'Printer')
            if printer_ctor is not None:
                printer = printer_ctor(args)
            else:
                print("Printer {} is not supported".format(printer_str))
                exit()

        if args.display != None:
            for disp_opt in args.display.split(','):
                if disp_opt == 'inventory':
                    printer.print_inventory( reduce_transforms.get_inventory(tplgy) )
                elif disp_opt == 'unique':
                    printer.print_unique_all( reduce_transforms.get_uniques_inventory(tplgy) )
                elif disp_opt == 'output':
                    print("outputs:")
                    outputs = tplgy.find_output_blobs()
                    for output in outputs:
                        print('\t' + output)
                elif disp_opt == 'bfs':
                    printer.print_bfs(tplgy)
                elif disp_opt == 'mem':
                    sum = [0]
                    blobs = []
                    tplgy.traverse(lambda node: sum_blob_mem(tplgy, node, blobs, sum))
                    print("Total BLOB memory: " + str(sum[0]))
                else:
                    exit ("Error: invalid display option")


if __name__ == '__main__':
    main()
