#!/bin/bash
# nvdisasm named in a comment is text
nvdisasm -cfg "$1" > out.dot
make -j4
d=$(mktemp -d)
