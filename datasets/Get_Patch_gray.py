import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms


def load_image(image_path, is_resize=True, mode='RGB'):
    if mode == 'RGB':
        img = Image.open(image_path).convert('RGB')
    else:
        img = Image.open(image_path).convert('L')
    if is_resize:
        t = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.ToTensor(),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        img = t(img)
    if mode == 'L':
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)
    else:
        img = np.array(img)
    return img


def is_valid_patch(patch, threshold=1):
    mean_value = np.mean(patch)
    return threshold < mean_value < (255 - threshold)


def get_distributed_random_patches(image1, image2, patch_size=(128, 128), mode='RGB'):
    #assert image1.shape == image2.shape, "Images must have the same dimensions"
    h, w, _ = image1.shape
    patch_h, patch_w = patch_size
    num_patches = int(max(h, w) / max(patch_h, patch_w))
    #print(num_patches)

    # 将图像划分为多个区域
    regions_per_dim = int(np.ceil(np.sqrt(num_patches)))
    region_h = h // regions_per_dim
    region_w = w // regions_per_dim

    patches_image1 = []
    patches_image2 = []

    for i in range(regions_per_dim):
        for j in range(regions_per_dim):
            if len(patches_image1) >= num_patches:
                break
            region_x_start = j * region_w
            region_y_start = i * region_h

            max_x = min(region_x_start + region_w - patch_w, w - patch_w)
            max_y = min(region_y_start + region_h - patch_h, h - patch_h)

            if max_x > region_x_start and max_y > region_y_start:
                x = random.randint(region_x_start, max_x)
                y = random.randint(region_y_start, max_y)

                patch1 = image1[y:y + patch_h, x:x + patch_w]
                patch2 = image2[y:y + patch_h, x:x + patch_w]

                if is_valid_patch(patch1) or is_valid_patch(patch2):
                    patches_image1.append(patch1)
                    patches_image2.append(patch2)
                    break

                # patches_image1.append(patch1)
                # patches_image2.append(patch2)

    # 如果生成的patches少于要求的数量，补齐剩余的patch
    while len(patches_image1) < num_patches:
        x = random.randint(0, w - patch_w)
        y = random.randint(0, h - patch_h)

        patch1 = image1[y:y + patch_h, x:x + patch_w]
        patch2 = image2[y:y + patch_h, x:x + patch_w]

        if is_valid_patch(patch1) or is_valid_patch(patch2):
            patches_image1.append(patch1)
            patches_image2.append(patch2)
            break

        # patches_image1.append(patch1)
        # patches_image2.append(patch2)

    return patches_image1, patches_image2


def EN(img):
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a) / a.shape[0]
    E = -sum(h * np.log2(h + (h == 0)))
    return E


def Get_Random_Patch(img1_path, img2_path, is_resize=True, patch_size=(128, 128), mode='gray'):
    image1 = load_image(img1_path, is_resize)
    image2 = load_image(img2_path, is_resize, mode='L')
    patches_image1, patches_image2 = get_distributed_random_patches(image1, image2, patch_size=patch_size)
    img_en = EN(image2)
    p = random.random()
    avl = []
    for i, (patch1, patch2) in enumerate(zip(patches_image1, patches_image2)):
        p2_en = EN(patch2)
        if p2_en > img_en:
            avl.append(i)
    random_index = random.randint(0, len(patches_image1) - 1)
    selected_patch1 = patches_image1[random_index]
    selected_patch2 = patches_image2[random_index]
    selected_patch2 = np.squeeze(selected_patch2, axis=-1)
    selected_patch2_rgb = np.stack([selected_patch2] * 3, axis=-1)

    patch1_img = Image.fromarray(selected_patch1)
    patch2_img = Image.fromarray(selected_patch2)
    patch2_img_rgb = Image.fromarray(selected_patch2_rgb)

    return patch1_img, patch2_img, patch2_img_rgb


# 示例用法
# image_path1 = '/home/yyan/zyx/MambaIR-UIF/image_fusion/train/MSRS/train/vi/00001D.png'  #/home/yyan/zyx/Fusion/Dataset/MSRS/test/vi/00095D.png
# image_path2 = '/home/yyan/zyx/MambaIR-UIF/image_fusion/train/MSRS/train/ir/00001D.png'
# patch1_img, patch2_img, patch2_img_rgb = Get_Random_Patch(img1_path=image_path1, img2_path=image_path2)
# patch1_img.save(f'/home/yyan/zyx/Fusion/Dataset/MSRS/test/patch_test_gray/patch1.png')
# patch2_img.save(f'/home/yyan/zyx/Fusion/Dataset/MSRS/test/patch_test_gray/patch2.png')
# patch2_img_rgb.save(f'/home/yyan/zyx/Fusion/Dataset/MSRS/test/patch_test_gray/patch2_img_rgb.png')

