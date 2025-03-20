#!/bin/bash

pytest pymedm -v -n auto -r a --color yes --cov pymedm --cov-report term-missing --doctest-modules