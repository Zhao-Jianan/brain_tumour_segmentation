import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_segmentation_on_modalities(t1_path, t1ce_path, t2_path, flair_path, pred_path, output_path, fig_title):
    # 1. 加载各模态图像（T1, T1ce, T2, FLAIR）
    t1_img = nib.load(t1_path)
    t1ce_img = nib.load(t1ce_path)
    t2_img = nib.load(t2_path)
    flair_img = nib.load(flair_path)
    
    t1_data = t1_img.get_fdata()
    t1ce_data = t1ce_img.get_fdata()
    t2_data = t2_img.get_fdata()
    flair_data = flair_img.get_fdata()

    # 2. 加载预测标签图像
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()

    # 3. 定义颜色映射
    label_colors = {
        0: [0, 0, 0, 0],       # Background: 黑色，透明
        1: [0, 1, 0, 1],       # Edema: 绿色
        2: [1, 1, 0, 1],       # Non-enhancing: 黄色
        3: [1, 0, 0, 1]        # Enhancing: 红色
    }

    # 4. 为每个标签区域赋颜色，生成 RGBA 图像
    rgba_img = np.zeros((pred_data.shape[0], pred_data.shape[1], pred_data.shape[2], 4))
    for label, color in label_colors.items():
        rgba_img[pred_data == label] = color

# 5. 选择肿瘤像素最多的切片
    tumor_pixel_counts = [np.sum(pred_data[:, :, i] > 0) for i in range(pred_data.shape[2])]

    if all(count == 0 for count in tumor_pixel_counts):
        slice_idx = pred_data.shape[2] // 2  # 所有像素都是0，选中间那一张
    else:
        slice_idx = int(np.argmax(tumor_pixel_counts))  # 否则选肿瘤最多的那一张

    # 6. 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

    # 7. 设置大标题和副标题
    fig.suptitle("Tumour Segmentation Visualisation", fontsize=30, y=1.12)
    fig.text(0.98, 1.05, f"{fig_title}", ha='right', fontsize=18)
    fig.text(0.98, 1.02, f"(slice: {slice_idx+1} / {pred_data.shape[2]})", ha='right', fontsize=18)

    # 8. 绘制原始模态图像
    modalities = [t1_data, t1ce_data, t2_data, flair_data]
    titles = ["T1", "T1ce", "T2", "FLAIR"]

    for i in range(4):
        axes[0, i].imshow(np.rot90(modalities[i][:, :, slice_idx]), cmap='gray')
        axes[0, i].set_title(titles[i], fontsize=18)
        axes[0, i].axis('off')

    # 9. 绘制带分割结果的图像
    for i in range(4):
        axes[1, i].imshow(np.rot90(modalities[i][:, :, slice_idx]), cmap='gray')
        axes[1, i].imshow(np.rot90(rgba_img[:, :, slice_idx]), alpha=1)
        axes[1, i].set_title(f"{titles[i]} with Segmentation", fontsize=18)
        axes[1, i].axis('off')

    # 10. 图例
    legend_labels = {
        'Edema': label_colors[1],
        'Necrosis': label_colors[2],
        'Enhancing Tumour': label_colors[3]
    }
    handles = [mpatches.Patch(color=color[:3], label=label) for label, color in legend_labels.items()]
    fig.legend(handles=handles, loc='upper center', fontsize=20, bbox_to_anchor=(0.5, 0), ncol=3)

    # 11. 保存和显示
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print(f"Subplots saved to {output_path}")
