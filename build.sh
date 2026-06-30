#!/bin/bash

CXX=g++
CXXFLAGS="-std=c++17 -O3 -I./include -Wall -Wextra"
TARGET="gisa_test"
SRC="src/main.cpp" # Updated path to reflect project structure

echo "Building GISA Engine..."

$CXX $CXXFLAGS $SRC -o $TARGET

if [ $? -eq 0 ]; then
    echo "Build successful: $TARGET"
    echo "Running integration test..."
    ./$TARGET
else
    echo "Build failed!"
    exit 1
fi