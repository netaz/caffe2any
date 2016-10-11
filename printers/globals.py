import caffe_pb2 as caffe

def get_pooling_types_dict():
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_eltwise_op_dict():
    desc = caffe.EltwiseParameter.EltwiseOp.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_lrn_types_dict():
    desc = caffe.LRNParameter.NormRegion.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d

lrn_type = get_lrn_types_dict()

#from printers.png import PngPrinter
from printers import csv, console

def get_printer(printer_name, param1):
    printer_list = {
        'csv': csv.CsvPrinter(fname = param1),
    }

    return printer_list[printer_name]