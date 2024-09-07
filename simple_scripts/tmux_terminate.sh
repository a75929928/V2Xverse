#!/bin/bash
# 找到监听端口 2000 的进程的 PID 并发送 SIGTERM 信号
PID=$(lsof -t -i:2000)
if [ -n "$PID" ]; then
    kill -SIGTERM "$PID"
fi

