#!/usr/bin/env python3
"""
将目录下的所有PNG图片转换为PDF格式
"""
import os
from PIL import Image

def convert_png_to_pdf(input_dir):
    """将目录下所有PNG文件转换为PDF"""
    # 获取目录下所有PNG文件
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print("未找到PNG文件")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    for png_file in png_files:
        png_path = os.path.join(input_dir, png_file)
        pdf_path = os.path.join(input_dir, png_file.replace('.png', '.pdf'))
        
        try:
            # 打开PNG图片
            img = Image.open(png_path)
            
            # 如果是RGBA模式，转换为RGB（PDF不支持透明度）
            if img.mode == 'RGBA':
                # 创建白色背景
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存为PDF
            img.save(pdf_path, 'PDF', resolution=100.0)
            print(f"✓ 已转换: {png_file} -> {os.path.basename(pdf_path)}")
            
        except Exception as e:
            print(f"✗ 转换失败 {png_file}: {e}")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_png_to_pdf(script_dir)
    print("\n转换完成！")

