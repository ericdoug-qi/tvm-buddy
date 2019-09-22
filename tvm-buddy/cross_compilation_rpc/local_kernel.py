#!/usr/bin/env python
#-*- coding: utf-8 -*-
# author: qi
# datetime: 2019-09-22 09:49
# software:


import numpy as np
import tvm

from tvm import rpc
from tvm.contrib import util


n = tvm.convert(1024)
A = tvm.placeholder((n, ), name='A')
B = tvm.compute((n,), lambda i: A[i] + 1.0, name='B')
s = tvm.create_schedule(B.op)

local_demo = False

if local_demo:
    target = 'llvm'
else:
    target = 'llvm -target=armv7l-linux-guneabihf'

func = tvm.build(s, [A, B], target=target, name='add_one')
# save the lib at a local temp folder
temp = util.tempdir()
path = temp.relpath('lib.tar')
func.export_library(path)

if local_demo:
    remote = rpc.LocalSession()
else:
    host = '192.168.0.166'
    port = 9090
    remote = rpc.connect(host, port)

remote.upload(path)
func = remote.load_module('lib.tar')

ctx = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.type), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)

func(a, b)

np.testing.assert_(b.asnumpy(), a.asnumpy() + 1)
time_f = func.time_evaluator(func.entry_name, ctx, number=10)
cost = time_f(a, b).mean
print('%g secs/op' % cost)



