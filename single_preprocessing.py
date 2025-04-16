import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import glob, subprocess, ants
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def show_slices(data, title=None, cmap='gray'):
    """可视化3个体位平面的切片"""
    fig, axes = plt.subplots(1, 3)
    slices = [data.shape[0]//2, data.shape[1]//2, data.shape[2]//2]
    axes[0].imshow(data[slices[0], :, :], cmap=cmap)
    axes[1].imshow(data[:, slices[1], :], cmap=cmap)
    axes[2].imshow(data[:, :, slices[2]], cmap=cmap)
    if title:
        fig.suptitle(title)
    plt.show()


def load_and_convert_to_ras(input_path):
    """加载图像并转换为 RAS 方向"""
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine

    current_ornt = nib.aff2axcodes(affine)
    print(f"  当前方向: {current_ornt}")

    if current_ornt != ('R', 'A', 'S'):
        print("  转换为 RAS 方向")
        ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        orig_ornt = nib.orientations.axcodes2ornt(current_ornt)
        transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)

        data = nib.orientations.apply_orientation(data, transform)
        affine = img.as_reoriented(transform).affine

        new_ornt = nib.aff2axcodes(affine)
        print(f"  转换后方向: {new_ornt}")
        
    else:
        print("  已经是 RAS，无需转换")

    return data, affine


def resample_volume(data, affine):
    """
    重采样函数（自动调整输出尺寸使体素为1.0x1.0x1.0mm）
    
    Args:
        data: 输入3D数组 (z,y,x)
        affine: 原affine矩阵 (4x4)
        
    Returns:
        resampled: 重采样后的数据
        new_affine: 更新后的affine矩阵（体素为1mm）
    """
    
    print(f"原始体素尺寸: {np.sqrt(np.sum(affine[:3,:3]**2, axis=0))} mm")
    
    # 计算当前物理空间体素尺寸 (单位：mm)
    original_voxel = np.sqrt(np.sum(affine[:3,:3]**2, axis=0))
    
    # 目标体素尺寸 [1.0, 1.0, 1.0] mm
    target_voxel = np.array([1.0, 1.0, 1.0])
    
    # 计算输出尺寸（基于物理尺寸比例）
    output_shape = np.round(np.array(data.shape) * original_voxel / target_voxel).astype(int)
    print(f"自动计算输出尺寸: {data.shape} -> {output_shape}")
    
    # 计算缩放因子（注意顺序：z,y,x）
    scaling_factors = output_shape / np.array(data.shape)
    
    # 执行重采样（三阶插值）
    resampled = zoom(data.astype(np.float32), scaling_factors, order=3)
    
    # 构建新的affine矩阵（保持旋转方向，设置体素为1mm）
    new_affine = affine.copy()
    
    # 归一化方向向量并设置1mm体素
    for i in range(3):
        direction = new_affine[:3, i]
        norm = np.linalg.norm(direction)
        if norm > 0:
            new_affine[:3, i] = (direction / norm) * target_voxel[i]
    
    # 调整平移量（保持物理中心不变）
    original_center = np.dot(affine, np.array([(s-1)/2 for s in data.shape] + [1]))[:3]
    new_center = np.dot(new_affine, np.array([(s-1)/2 for s in resampled.shape] + [1]))[:3]
    new_affine[:3, 3] += (original_center - new_center)
    
    
    # 验证输出体素尺寸
    final_voxel = np.sqrt(np.sum(new_affine[:3,:3]**2, axis=0))
    print(f"重采样完成。实际体素尺寸: {final_voxel} mm")
    
    return resampled, new_affine


def multimodal_alignment(moving_data, fixed_data, moving_affine, fixed_affine):
    """Align moving image to fixed image space"""
    print("Performing multimodal alignment...")
    
    # Convert affine translation parts to tuples
    fixed_origin = tuple(fixed_affine[:3, 3].tolist())  # Convert to list then tuple
    moving_origin = tuple(moving_affine[:3, 3].tolist())
    
    # Convert to ANTs images with proper origin format
    fixed_img = ants.from_numpy(
        fixed_data.astype('float32'), 
        origin=fixed_origin  
    )
    moving_img = ants.from_numpy(
        moving_data.astype('float32'), 
        origin=moving_origin  
    )
    
    # Perform registration (keeping original parameters)
    reg = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Rigid',  
        grad_step=0.2,
        reg_iterations=(40, 20, 0))
    
    # Apply transformation
    aligned = ants.apply_transforms(
        fixed=fixed_img,
        moving=moving_img,
        transformlist=reg['fwdtransforms'])
    
    return aligned.numpy(), fixed_affine


def pad_or_crop_to_target(data, target_shape):
    """
    将3D图像数据补零或裁剪到指定的 target_shape。

    参数：
        data (ndarray): 原始3D图像数据。
        target_shape (tuple): 目标尺寸 (x, y, z)。

    返回：
        padded_or_cropped_data (ndarray): 补零/裁剪后的数据。
    """
    current_shape = data.shape
    padding = []
    padded_or_cropped_data = data.copy()

    for i in range(3):
        diff = target_shape[i] - current_shape[i]
        if diff == 0:
            print(f"  维度 {i} 的尺寸相同，无需补零")
            padding.append((0, 0))
        elif diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            padding.append((pad_before, pad_after))
            print(f"  维度 {i} 补零: 前 {pad_before}，后 {pad_after}")
        else:
            # 如果目标维度小于当前维度，裁剪固定数量
            total_remove = 10
            if abs(diff) < total_remove:
                remove_before = abs(diff) // 2
                remove_after = abs(diff) - remove_before
            else:
                remove_before = total_remove // 2
                remove_after = total_remove - remove_before

            start = remove_before
            end = current_shape[i] - remove_after
            slicer = [slice(None)] * 3
            slicer[i] = slice(start, end)
            padded_or_cropped_data = padded_or_cropped_data[tuple(slicer)]
            print(f"  维度 {i} 裁剪: 前 {remove_before}，后 {remove_after}，裁剪后形状: {padded_or_cropped_data.shape}")

            # 再判断是否需要补零
            new_diff = target_shape[i] - padded_or_cropped_data.shape[i]
            if new_diff >= 0:
                pad_before = new_diff // 2
                pad_after = new_diff - pad_before
                padding.append((pad_before, pad_after))
                print(f"  维度 {i} 再补零: 前 {pad_before}，后 {pad_after}")
            else:
                raise ValueError(f"维度 {i} 裁剪后仍大于目标尺寸，超出预期")

    padded_or_cropped_data = np.pad(padded_or_cropped_data, padding, mode='constant', constant_values=0)
    print(f"  最终形状: {padded_or_cropped_data.shape}")
    return padded_or_cropped_data


def n4_bias_correction(input_data, affine):
    """使用 SimpleITK 进行 N4 偏置场校正"""

    print("使用 SimpleITK 进行 N4 校正...")

    # 将 numpy array 转换为 SimpleITK 图像
    print("Step 1: 转换为 SimpleITK 图像")
    sitk_image = sitk.GetImageFromArray(input_data.astype(np.float32))
    
    # 可选：设置 origin 和 spacing，如果 affine 中有这些信息
    # 这里简化处理，仅使用图像数据本身

    # 估计初始偏置场掩膜
    print("Step 2: 使用 Otsu 阈值生成掩膜")
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

    # 设置滤波器
    print("Step 3: 配置 N4 校正参数")
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([30, 30, 20])
    corrector.SetConvergenceThreshold(1e-7)
    
    # 执行 N4 校正
    print("Step 4: 执行 N4 校正中...")
    corrected_image = corrector.Execute(sitk_image, mask_image)
    print("Step 5: 校正完成")
    # 转换回 numpy
    result = sitk.GetArrayFromImage(corrected_image)

    return result, affine


def run_brain_extraction(input_path, output_prefix='./preprocessed/brain_mask'):
    """
    使用 HD-BET 执行脑提取，仅保留二值脑掩膜文件（mask）

    参数:
    - input_path (str): 输入的 3D 图像路径
    - output_prefix (str): 输出文件名前缀（无扩展名），最终输出为 output_prefix + '_mask.nii.gz'

    返回:
    - mask_path (str): 掩膜文件路径
    """
    
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    cmd = [
        'hd-bet',
        '-i', input_path,
        '-o', output_prefix,
        '-b', '0',         # 不保存 stripped image
        '-s', '1'          # 保留 segmentation mask
    ]
    
    try:
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("脑提取完成！")
    except subprocess.CalledProcessError as e:
        print(f"HD-BET 执行失败: {e}")

    # mask 默认输出路径
    mask_path = output_prefix + '_mask.nii.gz'
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"未找到生成的掩膜文件: {mask_path}")
    
    return mask_path


def run_preprocessing_pipeline(input_path, 
                               reference_path=None, 
                               output_dir='./preprocessed', 
                               output_filename='preprocessed',
                               skip_n4=True,
                               batch_process=True,
                               target_shape=(240, 240, 160)):
    """完整的预处理流程"""
    
    # 步骤1: 加载数据
    print("步骤1/7: 加载原始数据，检查并转换图像方向为 RAS")
    input_data, input_affine = load_and_convert_to_ras(input_path)
    show_slices(input_data, "Original Input Data (RAS)")

    # 步骤2: 重采样
    print("\n步骤2/7: 重采样模态")
    resampled, new_affine = resample_volume(input_data, input_affine)
    show_slices(resampled, "After Resampling")

    # 如果是 T1ce，保存临时参考
    temp_reference_path = None
    if reference_path is None:
        temp_reference_path = os.path.join(output_dir, "temp_reference_resampled.nii.gz")
        resampled_img = nib.Nifti1Image(resampled, new_affine)
        nib.save(resampled_img, temp_reference_path)

    # 步骤3: 配准
    print("\n步骤3/7: 多模态配准")
    if reference_path is None:
        print("  跳过配准步骤（当前是 T1ce 模态）")
        aligned_data, aligned_affine = resampled, new_affine
    else:
        print("  使用 T1ce 重采样结果作为基准进行配准")
        reference_img = nib.load(reference_path)
        reference_data, reference_affine = reference_img.get_fdata(), reference_img.affine
        aligned_data, aligned_affine = multimodal_alignment(resampled, reference_data, new_affine, reference_affine)

    show_slices(aligned_data, "After Registration")

    # 步骤4: 补零/裁剪到目标大小
    print("\n步骤4/7: 补零/裁剪到目标大小")
    padded_or_cropped_data = pad_or_crop_to_target(aligned_data, target_shape)
    show_slices(padded_or_cropped_data, "After Padding/Cropping")


    # 步骤5: N4校正
    if skip_n4:
        print("\n跳过步骤5/7: 跳过偏置场校正")
        corrected = padded_or_cropped_data
    else:
        print("\n步骤5/7: 偏置场校正")
        corrected, _ = n4_bias_correction(padded_or_cropped_data, aligned_affine)
        show_slices(corrected, "After N4 Correction")

    # 步骤6: 颅骨剥离
    print("\n步骤6/7: 颅骨剥离")
    corrected_img = nib.Nifti1Image(corrected, aligned_affine)
    temp_corrected_path = os.path.join(output_dir, f'{output_filename}_corrected_temp.nii.gz')
    nib.save(corrected_img, temp_corrected_path)
    mask_output_path = os.path.join(output_dir, output_filename)

    mask_path = run_brain_extraction(
        input_path=temp_corrected_path,
        output_prefix=mask_output_path)

    brain_mask_img = nib.load(mask_path)
    brain_mask = brain_mask_img.get_fdata()
    os.remove(temp_corrected_path)

    # 步骤7: 强度标准化
    print("\n步骤7/7: 强度标准化")
    if batch_process:
        print("  跳过强度标准化")
        normalized = corrected
    else:
        normalized = (corrected - corrected[brain_mask > 0].mean()) / corrected[brain_mask > 0].std()
        show_slices(normalized, "After Normalization")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{output_filename}.nii.gz')
    nib.save(nib.Nifti1Image(normalized, aligned_affine), output_path)

    print(f"\n预处理完成！结果保存在:\n{output_path}\n{mask_path}")
    return normalized, brain_mask, aligned_affine, temp_reference_path if reference_path is None else reference_path


# 覆盖图显示函数
def show_overlay(image, mask, title="Overlay"):
    """显示图像和掩模的覆盖图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 自动选择有脑组织的切片
    slice_x = np.where(mask.sum(axis=(1,2)) > mask.max()*10)[0].mean().astype(int)
    slice_y = np.where(mask.sum(axis=(0,2)) > mask.max()*10)[0].mean().astype(int)
    slice_z = np.where(mask.sum(axis=(0,1)) > mask.max()*10)[0].mean().astype(int)
    
    slices = [slice_x, slice_y, slice_z]
    planes = ['Axial', 'Coronal', 'Sagittal']
    
    for i, (sl, plane) in enumerate(zip(slices, planes)):
        if i == 0:
            img_slice = image[sl, :, :]
            mask_slice = mask[sl, :, :]
        elif i == 1:
            img_slice = image[:, sl, :]
            mask_slice = mask[:, sl, :]
        else:
            img_slice = image[:, :, sl]
            mask_slice = mask[:, :, sl]
        
        axes[i].imshow(img_slice, cmap='gray')
        axes[i].imshow(mask_slice, cmap='jet', alpha=0.3)
        axes[i].set_title(f"{plane} View") 
        axes[i].axis('off')
    
    fig.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

