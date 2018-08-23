#!/usr/bin/env sh
set -e

./build/Output/Bin/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
