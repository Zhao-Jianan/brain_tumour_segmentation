from single_preprocessing import *


def fuse_brain_masks(mask_paths, output_path, mode='intersection'):
    """
    Fuse 4 brain masks into a single mask and save to file.

    Parameters:
        mask_paths (list of str): List of 4 brain mask image paths (NIfTI format).
        output_path (str): Path to save the fused brain mask.
        mode (str): 'intersection' or 'union' to combine the masks.

    Returns:
        nib.Nifti1Image: The fused brain mask image.
    """
    assert len(mask_paths) == 4, "You must provide exactly 4 brain mask paths."

    # Load and binarize brain masks
    masks = [(nib.load(p).get_fdata() > 0) for p in mask_paths]

    if mode == 'intersection':
        fused_mask_data = masks[0] & masks[1] & masks[2] & masks[3]
    elif mode == 'union':
        fused_mask_data = masks[0] | masks[1] | masks[2] | masks[3]
    else:
        raise ValueError("mode must be either 'intersection' or 'union'")

    # Use the affine from the first mask
    affine = nib.load(mask_paths[0]).affine

    # Save fused mask
    fused_mask_img = nib.Nifti1Image(fused_mask_data.astype(np.uint8), affine)
    nib.save(fused_mask_img, output_path)
    print(f"Fused brain mask saved to: {output_path}")

    return fused_mask_img


def final_standardization_and_normalization(patient_id, condition_name, output_root_dir, fused_mask, type="std"):
    """
    对 MRI 图像进行标准化、归一化或两者都做，依据 type 参数。

    参数：
        patient_id (str): 病人 ID
        condition_name (str): 条件名称
        output_root_dir (str): 图像文件所在目录
        fused_mask (nib.Nifti1Image): 融合后的脑区掩模
        type (str): 处理方式，取值："st"、"norm"、"std_norm"
    
    返回：
        result_dict (dict): 处理后的 Nifti1Image 字典，key 为 modality 名称
    """
    modality_list = ["T1ce", "T1", "T2", "FLAIR"]
    result_dict = {}

    fused_mask_data = fused_mask.get_fdata()
    mask_indices = fused_mask_data > 0

    def standardize(data, mask):
        brain_data = data[mask]
        mean = brain_data.mean()
        std = brain_data.std()
        result = np.zeros_like(data)
        result[mask] = (brain_data - mean) / std
        return result

    def normalize(data, mask):
        brain_data = data[mask]
        min_val = brain_data.min()
        max_val = brain_data.max()
        result = np.zeros_like(data)
        result[mask] = (brain_data - min_val) / (max_val - min_val)
        return result

    for modality in modality_list:
        filename = f"{patient_id}_{condition_name}_{modality}.nii.gz"
        file_path = os.path.join(output_root_dir, filename)
        print('file_path:', file_path)

        if not os.path.exists(file_path):
            print(f"    [跳过] 没有找到预处理后的 {modality} 文件")
            continue

        img = nib.load(file_path)
        data = img.get_fdata()
        affine = img.affine

        if type == "std":
            processed_data = standardize(data, mask_indices)

        elif type == "norm":
            processed_data = normalize(data, mask_indices)

        elif type == "std_norm":
            standardized_data = standardize(data, mask_indices)
            processed_data = normalize(standardized_data, mask_indices)

        else:
            raise ValueError("type 参数只接受 'std'、'norm' 或 'std_norm'")

        result_dict[modality] = nib.Nifti1Image(processed_data, affine)

    return result_dict


# 筛选函数
def is_target_modality(filename):
    if "T1_AX_Gadovist_1mm" in filename:
        return "T1ce"
    elif "T1_SE_TRA" in filename or 'T1_IR_AX' in filename:
        return "T1"
    elif "T2_TSE_TRA" in filename:
        return "T2"
    elif "FLAIR" in filename:
        return "FLAIR"
    return None 


# 批处理函数
def preprocess_single_check(input_root_dir,
                            output_root_dir,
                            patient_id,
                            condition_name,
                            reference_mode="T1ce",
                            model='nnunet',
                            type='std',
                            target_shape=(240, 240, 160)):
    os.makedirs(output_root_dir, exist_ok=True)

    # 获取病人文件夹路径
    patient_folder = os.path.join(input_root_dir, patient_id)
    if not os.path.exists(patient_folder):
        print(f"[跳过] 找不到病人 {patient_id}")
        return

    print(f"\n=== 正在处理病人: {patient_id} ===")

    # 获取病人的特定条件文件夹（例如“术前”）
    condition_folder = os.path.join(patient_folder, condition_name)
    if not os.path.exists(condition_folder):
        print(f"[跳过] 找不到条件 {condition_name} 文件夹")
        return

    print(f"  === 处理条件: {condition_name} ===")

    # 获取该子文件夹内的所有 nii.gz 文件
    all_files = glob.glob(os.path.join(condition_folder, "*.nii.gz"))

    # 筛选模态并构建模态字典
    modality_dict = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        modality = is_target_modality(filename)
        if modality:
            modality_dict[modality] = file_path

    # 确保找到了参考模态 T1ce
    t1ce_path = modality_dict.get("T1ce")
    if not t1ce_path:
        print(f"[跳过] 没有找到 T1ce 模态作为参考")
        return

    # 先处理 T1ce 模态
    print(f"  === 处理参考模态: T1ce ===")
    t1ce_filename = os.path.basename(t1ce_path)
    t1ce_filename_noext = patient_id + '_' + condition_name + '_T1ce'
    print(f" t1ce_filename_noext: {t1ce_filename_noext}")    
    t1ce_output_path = os.path.join(output_root_dir, patient_id, condition_name, t1ce_filename)
    os.makedirs(os.path.dirname(t1ce_output_path), exist_ok=True)
    print(f"    处理参考模态文件: {t1ce_filename}")

    # 执行 T1ce 预处理
    preprocessed_reference, mask, affine, reference_path = run_preprocessing_pipeline(
        t1ce_path,
        reference_path=None,
        output_dir=output_root_dir,
        output_filename=t1ce_filename_noext,
        skip_n4=False,
        batch_process=True,
        target_shape=target_shape
    )

    # 记录所有mask路径
    mask_paths = [os.path.join(output_root_dir, f"{patient_id}_{condition_name}_T1ce_mask.nii.gz")]

    # 处理其他模态
    for modality, input_path in modality_dict.items():
        if modality == "T1ce":
            continue  # 跳过参考模态本身

        filename = os.path.basename(input_path)
        filename_noext = patient_id + '_' + condition_name + f'_{modality}'
        output_path = os.path.join(output_root_dir, filename)
        print(f"    处理模态: {modality} - {filename}")

        # 执行预处理，使用重采样后的 T1ce 作为参考进行对齐
        preprocessed, mask, affine, _ = run_preprocessing_pipeline(
            input_path,
            reference_path=reference_path,  # 使用预处理后的 T1ce 作为参考
            output_dir=output_root_dir,
            output_filename=filename_noext,
            skip_n4=False,
            batch_process=True,
            target_shape=target_shape
        )

        mask_paths.append(os.path.join(output_root_dir, f"{patient_id}_{condition_name}_{modality}_mask.nii.gz"))


    # === 合并4个mask ===
    print("  === 合并brain mask ===")
    fused_mask_path = os.path.join(output_root_dir, f"{patient_id}_{condition_name}_fused_mask.nii.gz")
    fused_mask = fuse_brain_masks(mask_paths, fused_mask_path, mode='intersection')

    # === 执行标准化 ===
    print("  === 标准化各模态图像 ===")
    normalized_images = final_standardization_and_normalization(
        patient_id, 
        condition_name, 
        output_root_dir, 
        fused_mask, 
        type=type
    )

    # === 保存标准化后的图像 ===
    print("  === 保存标准化后的图像 ===")
    for modality, nifti_img in normalized_images.items():
        if model == 'nnunet':
            # nnUNet 格式重命名
            modality_map = {
                'FLAIR': '0000',
                'T1': '0001',
                'T1ce': '0002',
                'T2': '0003'
            }
            modality_suffix = modality_map.get(modality, modality)
        else:
            # 保留原始模态名称
            modality_suffix = modality

        output_filename = f"masked_{patient_id}_{condition_name}_{modality_suffix}.nii.gz"
        output_path = os.path.join(output_root_dir, output_filename)
        nib.save(nifti_img, output_path)
        print(f"    标准化后图像已保存: {output_path}")

    # === 删除临时 mask 文件 ===
    print("  === 删除临时 mask 文件 ===")
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            os.remove(mask_path)
            print(f"    已删除: {mask_path}")

    # === 删除临时重采样参考文件 ===
    temp_reference_path = os.path.join(output_root_dir, "temp_reference_resampled.nii.gz")
    if os.path.exists(temp_reference_path):
        os.remove(temp_reference_path)
        print(f"    已删除: {temp_reference_path}")

    print(f"=== 病人 {patient_id} 条件 {condition_name} 全部完成 ===")


