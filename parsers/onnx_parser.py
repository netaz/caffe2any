from topology import Topology
import logging
logger = None

def log():
    global logger
    if logger == None:
        logger = logging.getLogger('parsers')
    return logger

def parse_onnx_net(caffe_net):
    """
    Create and populate a Topology object, based on a given ONNX protobuf network object
    """
    graph = Topology()

def main():
    # Read a Caffe prototxt file
    try:
        f = open(sys.argv[1], "rb")
        text_format.Parse(f.read(), net)
        f.close()
    except IOError:
        exit("Could not open file " + sys.argv[1])

    tplgy = parse_onnx_net(net)

if __name__ == '__main__':
    main()
