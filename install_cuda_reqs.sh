#!/bin/bash

conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia -y
