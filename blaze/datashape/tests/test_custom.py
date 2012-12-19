from blaze.datashape import *

w = TypeVar('w')
x = TypeVar('x')
y = TypeVar('y')
z = TypeVar('z')

n = TypeVar('n')

Quaternion = complex64*(z*y*x*w)
RGBA = Record(R=int16, G=int16, B=int16, A=int8)

def setUp():
    Type.register('Quaternion', Quaternion)
    Type.register('RGBA', RGBA)

def test_custom_type():
    p1 = datashape('800, 600, RGBA')
    assert p1[2] is RGBA

    # We want to build records out of custom type aliases
    p2 = datashape('Record(x=Quaternion, y=Quaternion)')

def test_custom_stream():
    p1 = datashape('Stream, RGBA')
