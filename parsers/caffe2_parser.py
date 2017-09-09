#from topology import Topology
import sys
from google.protobuf import text_format
import caffe2_pb2 as caffe2
import logging
logger = None

def log():
    global logger
    if logger == None:
        logger = logging.getLogger('parsers')
    return logger

def parse_caffe2_net(caffe2_net):
    """
    Create and populate a Topology object, based on a given ONNX protobuf network object
    """
    '''
    proto = net if isinstance(net, caffe2_pb2.NetDef) else net.Proto()
    predict_net = caffe2_pb2.NetDef()
    predict_net.CopyFrom(proto)
    init_net = caffe2_pb2.NetDef()
    '''
    print(caffe2_net.name, caffe2_net.type)
    #graph = Topology()
    for op in caffe2_net.op:
        print(op.type)
        for arg in op.arg:
            if arg.HasField('name'): print('\t' + arg.name)

# python caffe2_parser.py ../examples/caffe2/alexnet_predict_net.pb

def main():
    net = caffe2.NetDef()
    # Read a Caffe2 prototxt file
    try:
        f = open(sys.argv[1], "rb")
        net.ParseFromString(f.read())
        # print(net) - use this for debugging
        f.close()
    except IOError:
        exit("Could not open file " + sys.argv[1])

    tplgy = parse_caffe2_net(net)

if __name__ == '__main__':
    main()
