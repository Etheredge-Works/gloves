#! /bin/bash
echo $1
echo $2
mkdir -p $1 ; tar xzf $2 -C $1 --strip-components 1