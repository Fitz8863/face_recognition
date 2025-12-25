#!/bin/bash

# convert_to_jpg.sh
# 将当前目录下所有非JPEG图片（如WebP、PNG等）转换为标准 .jpg 格式
# 输出文件名为: 原文件名.jpg （如果原文件是 a.webp → a.jpg）

set -e  # 遇错退出

echo "🔍 扫描当前目录中的图片..."

for file in *; do
    # 跳过非文件（如目录）
    [ -f "$file" ] || continue

    # 获取真实 MIME 类型
    mime_type=$(file --mime-type -b "$file")
    filename="${file%.*}"      # 去掉扩展名
    ext_lower="${file##*.}"
    ext_lower=$(echo "$ext_lower" | tr '[:upper:]' '[:lower:]')

    case "$mime_type" in
        image/jpeg)
            echo "✅ '$file' 已是 JPEG，跳过。"
            ;;
        image/webp)
            echo "🔄 转换 WebP: $file → $filename.jpg"
            dwebp "$file" -o "/tmp/${filename}.png" 2>/dev/null
            convert "/tmp/${filename}.png" -quality 95 "${filename}.jpg"
            rm -f "/tmp/${filename}.png"
            ;;
        image/png)
            echo "🔄 转换 PNG: $file → $filename.jpg"
            convert "$file" -quality 95 "${filename}.jpg"
            ;;
        image/gif|image/bmp|image/tiff)
            echo "🔄 转换 $mime_type: $file → $filename.jpg"
            convert "$file" -quality 95 "${filename}.jpg"
            ;;
        *)
            echo "⚠️  跳过非图片文件: $file ($mime_type)"
            ;;
    esac
done

echo "✨ 转换完成！"