#!/bin/bash
echo "Python interpreter: $(which python)"
echo "Mypy version: $(mypy --version)"
exec mypy "$@"
