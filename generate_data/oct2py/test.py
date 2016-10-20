#!/usr/bin/env python
'''test oct2py
https://pypi.python.org/pypi/oct2py
'''

from oct2py import octave
import os

octave.addpath(os.getcwd())
out = octave.test_oct2py(1)
print(out)
