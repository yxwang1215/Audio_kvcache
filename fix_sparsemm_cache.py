#!/usr/bin/env python3
"""
批量修复 SparseMM 两次 generate 调用的 cache 污染问题

问题: 第一次 generate 调用后 past_key_values 被污染,
      第二次 generate 调用时复用了污染的 cache 导致输出错误

修复: 在两次 generate 调用之间添加:
      1. reset_sparsemm_state(model)
      2. past_key_values = DynamicCacheSplitHeadFlatten()
"""

import os
import re
import sys
import glob

# 要添加的修复代码
FIX_CODE = """
            # CRITICAL FIX: Reset SparseMM state and create fresh cache for second generate
            # The first generate call pollutes the cache state, so we MUST reset before the actual generation
            if USE_SPARSEKV and sparsemm_enabled:
                reset_sparsemm_state(model)
                past_key_values = DynamicCacheSplitHeadFlatten()  # Create fresh cache
"""

# 匹配模式: 在 prefill_end_event.record() 后面, Measure decode time 注释前面
PATTERN = re.compile(
    r'(            prefill_end_event\.record\(\)\s*\n)'
    r'(\s*\n?\s*# Measure decode time)',
    re.MULTILINE
)

def fix_file(filepath):
    """修复单个文件"""
    if not os.path.exists(filepath):
        return "not_found"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修复过
    if "CRITICAL FIX: Reset SparseMM state" in content:
        return "already_fixed"
    
    # 检查是否有需要修复的模式
    if not PATTERN.search(content):
        return "pattern_not_found"
    
    # 应用修复
    new_content = PATTERN.sub(r'\1' + FIX_CODE + r'\2', content)
    
    if new_content == content:
        return "no_change"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return "fixed"

def find_all_files(base_dir):
    """查找所有需要修复的 Python 文件"""
    patterns = [
        os.path.join(base_dir, "qwen2_5", "**", "*.py"),
        os.path.join(base_dir, "qwen3", "**", "*.py"),
        os.path.join(base_dir, "gemma", "**", "*.py"),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    return all_files

def main():
    print("=" * 60)
    print("SparseMM Cache 污染问题批量修复脚本")
    print("=" * 60)
    
    base_dir = "/mnt/users/wangyuxuan-20250915/Audio/Audio_kvcache"
    
    # 查找所有 Python 文件
    all_files = find_all_files(base_dir)
    print(f"\n找到 {len(all_files)} 个 Python 文件")
    
    fixed_count = 0
    already_fixed_count = 0
    pattern_not_found_count = 0
    
    for filepath in all_files:
        # 获取相对路径用于显示
        rel_path = os.path.relpath(filepath, base_dir)
        
        result = fix_file(filepath)
        
        if result == "fixed":
            print(f"  ✅ 已修复: {rel_path}")
            fixed_count += 1
        elif result == "already_fixed":
            print(f"  ✓ 已修复过: {rel_path}")
            already_fixed_count += 1
        elif result == "pattern_not_found":
            # 静默跳过不需要修复的文件
            pattern_not_found_count += 1
        elif result == "not_found":
            print(f"  ❌ 文件不存在: {rel_path}")
        elif result == "no_change":
            print(f"  ⚠ 修复未生效: {rel_path}")
    
    print("\n" + "=" * 60)
    print(f"结果统计:")
    print(f"  - 新修复: {fixed_count} 个文件")
    print(f"  - 已修复过: {already_fixed_count} 个文件")
    print(f"  - 不需要修复: {pattern_not_found_count} 个文件")
    print("=" * 60)

if __name__ == "__main__":
    main()
