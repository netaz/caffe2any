'''
py.test tests/tplgy_test.py
http://pythontesting.net/framework/pytest/pytest-introduction/
'''
import os
import filecmp

def test_alexnet_csv():
    ret = os.system("python caffe2any.py examples/alexnet.deploy.prototxt -p csv -d bfs")
    assert ret==0
    ret = filecmp.cmp("examples/alexnet.deploy.prototxt.csv", "tests/golden/alexnet.deploy.prototxt.csv")
    assert ret==True

def test_googlenet_v1_csv():
    ret = os.system("python caffe2any.py examples/googlenetv1.deploy.prototxt -p csv -d bfs")
    assert ret==0
    ret = filecmp.cmp("examples/googlenetv1.deploy.prototxt.csv", "tests/golden/googlenetv1.deploy.prototxt.csv")
    assert ret==True
