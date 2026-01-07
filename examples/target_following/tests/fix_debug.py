#!/usr/bin/env python3
"""批量修复 DEBUG 日志"""

import re

with open('test_gesture_following.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
modified = 0
for i, line in enumerate(lines):
    # 检查是否是 DEBUG 日志但不在 DEBUG_VERBOSE 条件下
    if 'print(f"[DEBUG]' in line and 'DEBUG_VERBOSE' not in line:
        # 获取当前缩进
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent
        # 包装成 DEBUG_VERBOSE 条件
        new_line = indent_str + 'if DEBUG_VERBOSE: ' + line.lstrip()
        new_lines.append(new_line)
        modified += 1
    else:
        new_lines.append(line)

with open('test_gesture_following.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
    
print(f'已修改 {modified} 处 DEBUG 日志')
