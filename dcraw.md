# dcraw
> 下载并安装[dcraw](https://www.dechifro.org/dcraw/)。

### 1. HDR imaging

#### Develop RAW images
dcraw -n 100 -w -o 1 -q 3 -T4 exposure*.nef

**-n 100**：利用微波法消除杂讯的同时保存影像细节。杂讯消除临界值建议使用100至1000之间的数值。

**-w **：使用相机所指定的白平衡值。 如果在档案中找不到此项资料，显示警告信息并改用其他方式调整白平衡。

**-o 1**：sRGB D65 (预设值) 。

**-q 3**：使用 Adaptive Homogeneity-Directed (AHD) 内插法来进行影像的解码。

**-T**：输出 TIFF 格式（附元数据）的影像档案。

**-4**：输出 16 位元线性档案（固定全白色值，不改变 gamma 值）。
