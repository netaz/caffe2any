'''
Use this script to download protobuf files, and to compile them.
'''
import requests
import tqdm
import zipfile
import os.path
import stat
from subprocess import call

proto_files = [
    "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto",
    "https://raw.githubusercontent.com/caffe2/caffe2/master/caffe2/proto/caffe2.proto",
    "https://raw.githubusercontent.com/onnx/onnx/master/onnx/onnx.proto" ]

def download_and_install_protoc():
    """ This downloads and install's Google's protobuf compiler.

    You don't need this any longer, if you've followed DEPENDENCIES.md and
    installed protoc through Anaconda:
        conda install -c anaconda protobuf
    I'm keeping the code for completeness.
    """
    # Download Google's protobuf compiler
    protoc = 'https://github.com/google/protobuf/releases/download/v3.3.0/'
    protoc_version = 'protoc-3.3.0-linux-x86_64.zip'
    reply = requests.get(protoc + protoc_version, stream=True)
    print("Downloading ", protoc_version)
    with open(protoc_version, 'wb') as install_file:
        total_length = int(reply.headers.get('content-length'))

        for chunk in tqdm.tqdm(reply.iter_content(chunk_size=1024), total=(total_length+1023)//1024, unit='KB'):
            if chunk:
                install_file.write(chunk)
                install_file.flush()
        install_file.close()

    # Install protoc
    print("\nInstalling ", protoc_version)
    os.makedirs('protoc', exist_ok=True)
    zip = zipfile.ZipFile(protoc_version)
    zip.extractall('protoc')
    st = os.stat(protoc_version)
    os.chmod('protoc/bin/protoc', st.st_mode | stat.S_IEXEC)
    os.remove(protoc_version)
    return 'protoc/bin/protoc'

def download_and_compile_protos(proto_files, protoc):
    """ Compile the protobuf files """
    os.makedirs('protos', exist_ok=True)
    for link in proto_files:
        proto_content = requests.get(link)
        proto_file = 'protos/' + os.path.basename(link)
        print('\nDownloading ', proto_file)
        f = open(proto_file, 'w')
        f.write(proto_content.text)
        f.close()
        print("Compiling ", proto_file)
        call([protoc, '-I=.', '--python_out=.', proto_file])

def main():
    if False:
        # See deprecation notice above
        protoc = download_and_install_protoc()
    else:
        protoc = 'protoc'
    download_and_compile_protos(proto_files, protoc)

if __name__ == '__main__':
    main()
