SAH + FFT 参数敏感性绘图伪代码

```text
输入:
  - data: shape = [len(alpha_values), len(cutoff_values_all)] 的精度矩阵
  - alpha_values = 0, 0.1, ..., 1.0
  - cutoff_values_all = 0, 0.1, ..., 1.0
  - baseline_value = 91.36 (AudioKV w/o SSS 基线精度)

步骤:
1. 预处理数据
   1.1 拷贝一份 data 为 data_cleaned
   1.2 将 data_cleaned 中所有 < 85 的元素置为 NaN (用于去掉离群点, 不参与绘图)

2. 选择需要展示的 cutoff ratio 维度
   2.1 定义 indices_to_keep = [2, 3, 4, 5, 6, 7, 8, 9]  // 对应 cutoff = 0.2 ~ 0.9
   2.2 data_cleaned = data_cleaned[:, indices_to_keep]
   2.3 cutoff_values = cutoff_values_all[indices_to_keep]

3. 构造绘图网格
   3.1 使用 meshgrid:
       X, Y = meshgrid(cutoff_values, alpha_values)
       // X 轴: cutoff ratio
       // Y 轴: alpha

4. 计算 Z 轴范围
   4.1 min_val = nanmin(data_cleaned)
   4.2 max_val = nanmax(data_cleaned)
   4.3 z_min = max(87, min_val - 1)
   4.4 z_max = min(96, max_val + 1)

5. 创建 3D 图 (表面 + 等高线 + 基线平面)
   5.1 新建 figure 和 3D 坐标轴 ax
   5.2 选择 colormap = plasma
   5.3 绘制表面:
       surf = ax.plot_surface(
           X, Y, data_cleaned,
           cmap = colormap,
           alpha = 0.95,
           linewidth = 0.05,
           antialiased = True,
           edgecolor = "none",
           shade = True,
           rstride = 1,
           cstride = 1
       )
   5.4 在 z = z_min 平面上绘制等高线投影:
       ax.contour(
           X, Y, data_cleaned,
           zdir = "z",
           offset = z_min,
           cmap = colormap,
           alpha = 0.4,
           linewidths = 1.0,
           levels = 15
       )
   5.5 绘制 AudioKV 基线平面 (不含 FFT):
       baseline_Z = full_like(X, baseline_value)
       ax.plot_surface(
           X, Y, baseline_Z,
           alpha = 0.25,
           color = "gray",
           shade = False
       )

6. 在基线平面上绘制虚线网格 (辅助视觉)
   6.1 沿 alpha 方向 (X 轴方向画线):
       对每个 alpha_values[i]:
         在 z = baseline_value 处, 画一条 cutoff_values 上的虚线
   6.2 沿 cutoff 方向 (Y 轴方向画线):
       对每个 cutoff_values[j]:
         在 z = baseline_value 处, 画一条 alpha_values 上的虚线

7. 为基线添加文本标注
   7.1 在最大 cutoff 附近 (cutoff = max(cutoff_values), alpha = 1.0, z = baseline_value)
       画一条指向右侧的粗黑线
   7.2 在线的末端使用 text 在 3D 空间写:
       "AudioKV\n(w/o SSS)"
       并加粗、加边框、高亮背景

8. 设置视角和坐标轴
   8.1 ax.view_init(elev=35, azim=50)
   8.2 X 轴标签: "Cutoff Ratio", 粗体
   8.3 Y 轴标签: "Alpha", 粗体
   8.4 不设置 Z 轴标题 (用 colorbar 表示精度)
   8.5 手动设置刻度:
       - X: [0.2, 0.4, 0.6, 0.8]
       - Y: [0.0, 0.2, 0.4, 0.6, 0.8]
       - Z: 从 z_min 到 z_max, 步长 3
   8.6 设置坐标轴范围:
       - xlim: [min(cutoff_values) - 0.05, max(cutoff_values) + 0.05]
       - ylim: [0, 1.0]
       - zlim: [z_min, z_max]
   8.7 将所有刻度字体加粗

9. 添加 colorbar
   9.1 使用 surf 作为 mappable:
       cbar = fig.colorbar(
           surf,
           ax = ax,
           shrink = 0.7,
           aspect = 25,
           pad = 0.05
       )
   9.2 设置标签:
       "Accuracy (%)", 旋转 270°, 粗体
   9.3 限制 colorbar 刻度数量 (最多 5 个), 调整刻度字体并加粗

10. 风格与背景设置
    10.1 关闭三个坐标平面的填充, 仅保留浅灰色边框
    10.2 将三维背景设置为白色
    10.3 调整布局使用紧凑排版 (tight_layout)

11. 保存图像到本地
    11.1 output_dir = 当前脚本所在目录
    11.2 依次保存:
         - PNG:  "sah_fft_accuracy_heatmap.png"
         - SVG:  "sah_fft_accuracy_heatmap.svg"
         - PDF:  "sah_fft_accuracy_heatmap.pdf"
    11.3 每次保存后在终端打印文件路径 (用于日志和检查)

输出:
  - SAH + FFT 在不同 alpha 和 cutoff ratio 下的 3D 精度可视化图 (PNG/SVG/PDF)
  - 其中灰色平面表示不使用 FFT 的 AudioKV 基线, 用于对比提升效果
```


