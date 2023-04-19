#!/bin/bash

if [ -d out ]; then
    rm -rf out
fi

mkdir out
cd out || exit

if [ -f "Makefile" ]; then
  make clean
fi

if [ $MS_LITE_HOME ];then
  MINDSPORE_PATH=$MS_LITE_HOME/runtime
else
  MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  if [[ ! $MINDSPORE_PATH ]];then
      MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
  fi
fi
cmake .. -DMINDSPORE_PATH=$MINDSPORE_PATH
make
