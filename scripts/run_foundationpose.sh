#!/usr/bin/env bash
# 1) 讓系統庫優先，並把 conda 的 lib 路徑移到後面或剔除
SYS_LIB=/usr/lib/x86_64-linux-gnu

# 拆解 LD_LIBRARY_PATH，移除含 "conda" 的段落（保守作法：不移除，只把系統庫擺最前）
CLEAN_LD=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | awk '!/conda/' | paste -sd:)

# 2) 重新組合，並將系統庫放到最前面
export LD_LIBRARY_PATH="${SYS_LIB}:${CLEAN_LD}"

# （可選）如果你確定系統有 libffi.so.7，也能顯式預載，但要在 exec 前設定才有效
# export LD_PRELOAD="${SYS_LIB}/libffi.so.7"

# 3) 不需要 conda activate，因為 shebang 指向 conda python
#    直接執行你的節點腳本
exec /home/user/ros_fp/src/ros_foundationpose/scripts/test.py "$@"
