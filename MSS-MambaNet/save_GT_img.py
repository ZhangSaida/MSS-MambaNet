import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import argparse
from matplotlib.patches import Patch
from scipy import io

# 数据集名称到imageID的映射
DATASET_TO_ID = {
    'pu': 1,  # Pavia University
    'sa': 2,  # Salinas
    'houston13': 3,  # Houston2013
    'ip': 4,  # Indian_pines
    'whulk': 5,  # LongKou
    'whuhc': 6,  # HanChuan
    'whuhh': 7,  # HongHu
    'houston18': 8,  # Houston2018
    'QUH_TDW': 10,  # QUH_TDW
    'QUH_QY': 11,  # QUH_QY
    'QUH_PA': 12  # QUH_PA
}


def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'pu',
        'sa',
        'ip',
        'whulk',
        'whuhc',
        'whuhh',
        'houston13',
        'houston18',
        'QUH_TDW',
        'QUH_QY',
        'QUH_PA'
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if (dataset_name == 'sa'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'pu'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'ip'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_corrected.mat"))
        image = image['indian_pines_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_gt.mat"))
        gt = gt['indian_pines_gt']
        labels = [
            'Undefined',
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif (dataset_name == 'whulk'):
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhh':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu.mat"))
        image = image['WHU_Hi_HongHu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu_gt.mat"))
        gt = gt['WHU_Hi_HongHu_gt']
        labels = [
            'Undefined',
            'Red roof',
            'Road',
            'Bare soil',
            'Cotton',
            'Cotton firewood',
            'Rape',
            'Chinese cabbage',
            'Pakchoi',
            'Cabbage',
            'Tuber mustard',
            'Brassica parachinensis',
            'Brassica chinensis',
            'Small Brassica chinensis',
            'Lactuca sativa',
            'Celtuce',
            'Film covered lettuce',
            'Romaine lettuce',
            'Carrot',
            'White radish',
            'Garlic sprout',
            'Broad bean',
            'Tree',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhc':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan.mat"))
        image = image['WHU_Hi_HanChuan']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan_gt.mat"))
        gt = gt['WHU_Hi_HanChuan_gt']
        labels = [
            'Undefined',
            'Strawberry',
            'Cowpea',
            'Soybean',
            'Sorghum',
            'Water spinach',
            'Watermelon',
            'Greens',
            'Trees',
            'Grass',
            'Red roof',
            'Gray roof',
            'Plastic',
            'Bare soil',
            'Road',
            'Bright object',
            'Water',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'houston13':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013.mat"))
        image = image['GRSS2013']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "GRSS2013_gt.mat"))
        gt = gt['GRSS2013_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Synthetic grass",
            "Trees",
            "Soil",
            "Water",
            "Residential",
            "Commercial",
            "Road",
            "Highway",
            "Railway",
            "Parking Lot 1",
            "Parking Lot 2",
            "Tennis Court",
            "Running Track",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'houston18':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Houston2018.mat"))
        image = image['Houston2018']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Houston2018_gt.mat"))
        gt = gt['Houston2018_gt']
        labels = [
            "Undefined",
            "Healthy grass",
            "Stressed grass",
            "Artificial turf",
            "Evergreen trees",
            "Deciduous trees",
            "Bare earth",
            "Water",
            "Residential buildings",
            "Non-residential buildings",
            "Roads",
            "Sidewalks",
            "Crosswalks",
            "Major thoroughfares",
            "Highways",
            "Railways",
            "Paved parking lots",
            "Unpaved parking lots",
            "Cars",
            "Trains",
            "Stadium seats",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_TDW':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Tangdaowan.mat"))
        image = image['Tangdaowan']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Tangdaowan_GT.mat"))
        gt = gt['TangdaowanGT']
        labels = [
            "Undefined",
            "Rubber track",
            "Flagging",
            "Sandy",
            "Asphalt",
            "Boardwalk",
            "Rocky shallows",
            "Grassland",
            "Bulrush",
            "Gravel road",
            "Ligustrum vicaryi",
            "Coniferous pine",
            "Spiraea",
            "Bare soil",
            "Buxus sinica",
            "Photinia serrulata",
            "Populus",
            "Ulmus pumila L",
            "Seawater",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_QY':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Qingyun.mat"))
        image = image['Chengqu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Qingyun_GT.mat"))
        gt = gt['ChengquGT']
        labels = [
            "Undefined",
            "Trees",
            "Concrete building",
            "Car",
            "Ironhide building",
            "Plastic playground",
            "Asphalt road",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'QUH_PA':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Pingan.mat"))
        image = image['Haigang']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "QUH-Pingan_GT.mat"))
        gt = gt['HaigangGT']
        labels = [
            "Undefined",
            "Ship",
            "Seawater",
            "Trees",
            "Concrete structure building",
            "Floating pier",
            "Brick houses",
            "Steel houses",
            "Wharf construction land",
            "Car",
            "Road",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0


    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 filter nan values out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # step2 normalise the HSI data (method from SSAN, TGRS 2020)
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    mean_by_c = np.mean(image, axis=(0, 1))
    for c in range(image.shape[-1]):
        image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    return image, gt, labels


def get_palette_for_dataset(imageID):
    """获取指定数据集的调色板"""
    if imageID == 1:  # Pavia University
        palette = np.array([
            [216, 191, 216], [0, 255, 0], [0, 255, 255],
            [45, 138, 86], [255, 0, 255], [255, 165, 0],
            [159, 31, 239], [255, 0, 0], [255, 255, 0]
        ]) / 255.0

    elif imageID == 2:  # Salinas
        palette = np.array([
            [37, 58, 150], [47, 78, 161], [56, 87, 166],
            [56, 116, 186], [51, 181, 232], [112, 204, 216],
            [119, 201, 168], [148, 204, 120], [188, 215, 78],
            [238, 234, 63], [246, 187, 31], [244, 127, 33],
            [239, 71, 34], [238, 33, 35], [180, 31, 35],
            [123, 18, 20]
        ]) / 255.0

    elif imageID == 3:  # Houston2013
        palette = np.array([
            [0, 205, 0], [127, 255, 0], [46, 139, 87],
            [0, 139, 0], [160, 82, 45], [0, 255, 255],
            [255, 255, 255], [216, 191, 216], [255, 0, 0],
            [139, 0, 0], [0, 0, 0], [255, 255, 0],
            [238, 154, 0], [85, 26, 139], [255, 127, 80]
        ]) / 255.0

    elif imageID == 4:  # Indian_pines
        palette = np.array([
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [176, 48, 96], [46, 139, 87], [160, 32, 240],
            [255, 127, 80], [127, 255, 212], [218, 112, 214],
            [160, 82, 45], [127, 255, 0], [216, 191, 216],
            [238, 0, 0]
        ]) / 255.0

    elif imageID == 5:  # LongKou
        palette = np.array([
            [255, 0, 0], [239, 155, 0], [255, 255, 0],
            [0, 255, 0], [0, 255, 255], [0, 140, 140],
            [0, 0, 255], [255, 255, 255], [160, 32, 240]
        ]) / 255.0

    elif imageID == 6:  # HanChuan
        palette = np.array([
            [176, 48, 96], [0, 255, 255], [255, 0, 255],
            [160, 32, 240], [127, 255, 212], [127, 255, 0],
            [0, 205, 0], [0, 255, 0], [0, 139, 0],
            [255, 0, 0], [216, 191, 216], [255, 127, 80],
            [160, 82, 45], [255, 255, 255], [218, 112, 214],
            [0, 0, 255]
        ]) / 255.0

    elif imageID == 7:  # HongHu
        palette = np.array([
            [255, 0, 0], [255, 255, 255], [176, 48, 96],
            [255, 255, 0], [255, 127, 80], [0, 255, 0],
            [0, 205, 0], [0, 139, 0], [127, 255, 212],
            [160, 32, 240], [216, 191, 216], [0, 0, 255],
            [0, 0, 139], [218, 112, 214], [160, 82, 45],
            [0, 255, 255], [255, 165, 0], [127, 255, 0],
            [139, 139, 0], [0, 139, 139], [205, 181, 205],
            [238, 154, 0]
        ]) / 255.0

    elif imageID == 8:  # Houston2018
        palette = np.array([
            [0, 206, 0], [123, 220, 0], [47, 139, 85],
            [0, 136, 0], [0, 68, 0], [159, 79, 41],
            [68, 227, 251], [255, 255, 255], [213, 191, 213],
            [248, 2, 0], [167, 160, 146], [124, 124, 124],
            [160, 4, 3], [80, 0, 5], [226, 161, 13],
            [255, 242, 3], [237, 153, 0], [242, 0, 200],
            [0, 6, 191], [172, 196, 219]
        ]) / 255.0

    elif imageID == 10:  # QUH_TDW
        palette = np.array([
            [140, 67, 46], [153, 153, 153], [255, 100, 0],
            [0, 255, 123], [164, 75, 155], [101, 174, 255],
            [118, 254, 172], [60, 91, 112], [255, 255, 0],
            [255, 255, 125], [255, 0, 255], [100, 0, 255],
            [0, 172, 254], [0, 255, 0], [171, 175, 80],
            [101, 193, 60], [139, 0, 0], [0, 0, 255]
        ]) / 255.0

    elif imageID == 11:  # QUH_QY
        palette = np.array([
            [0, 255, 0], [153, 153, 153], [255, 100, 0],
            [164, 75, 155], [101, 174, 255], [140, 67, 46]
        ]) / 255.0

    elif imageID == 12:  # QUH_PA
        palette = np.array([
            [140, 67, 46], [0, 0, 255], [0, 200, 0],
            [101, 174, 255], [164, 75, 155], [192, 80, 70],
            [60, 91, 112], [255, 255, 0], [255, 100, 0],
            [118, 254, 172]
        ]) / 255.0

    else:
        # 默认调色板 (使用tab20颜色映射)
        cmap = plt.cm.get_cmap('tab20', 20)
        palette = cmap(np.arange(20))[:, :3]  # 只取RGB，忽略alpha

    return palette


def draw_gt(gt, imageID, labels):
    """绘制GT标签图像"""
    # 获取数据集对应的调色板
    palette = get_palette_for_dataset(imageID)

    # 创建结果图像
    h, w = gt.shape
    result = np.zeros((h, w, 3))

    # 获取唯一标签值
    unique_labels = np.unique(gt)
    num_classes = len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)

    # 为每个标签分配颜色
    for i, label in enumerate(unique_labels):
        if label == -1:  # 未定义区域
            color = [0, 0, 0]  # 黑色
        else:
            # 使用循环调色板，以防类别数超过调色板长度
            color_idx = label % len(palette)
            color = palette[color_idx]

        result[gt == label] = color

    # 创建图例元素
    legend_elements = []
    for i, label in enumerate(unique_labels):
        if label == -1:
            name = "Undefined"
            color = [0, 0, 0]
        else:
            name = labels[label] if label < len(labels) else f"Class {label}"
            color_idx = label % len(palette)
            color = palette[color_idx]

        legend_elements.append(Patch(facecolor=color, label=name))

    # 绘制图像
    plt.figure(figsize=(12, 10))
    plt.imshow(result)
    plt.title(f'Ground Truth (Dataset ID: {imageID})')
    plt.axis('off')

    # 添加图例
    plt.legend(handles=legend_elements,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               title="Classes")

    return result, legend_elements


def plot_false_color(image, dataset_name):
    """绘制HSI伪彩色图像"""
    #选择三个波段作为RGB(可以根据数据集调整波段)
    if image.shape[2] > 100:  # 高光谱数据
        r_band = min(60, image.shape[2] - 1)
        g_band = min(30, image.shape[2] - 1)
        b_band = min(10, image.shape[2] - 1)
    else:  # 多光谱数据
        r_band = min(2, image.shape[2] - 1)
        g_band = min(1, image.shape[2] - 1)
        b_band = min(0, image.shape[2] - 1)

    # 提取RGB波段
    rgb_bands = [r_band, g_band, b_band]
    rgb_img = image[:, :, rgb_bands]

    # 归一化
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))

    # 绘制图像
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_img)
    plt.title(f'False Color Image of {dataset_name} Dataset (Bands: {rgb_bands})')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉四周边距

    return rgb_img


def save_visualizations(dataset_name, dataset_dir, output_dir='results'):
    """保存HSI和GT的可视化图像"""
    # 创建输出目录
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # 获取数据集ID
    imageID = DATASET_TO_ID.get(dataset_name, 0)

    # 加载数据
    image, gt, labels = load_mat_hsi(dataset_name, dataset_dir)

    # 1. 保存HSI伪彩色图像
    plt.figure()
    rgb_img = plot_false_color(image, dataset_name)
    rgb_path = os.path.join(output_dir, f'{dataset_name}_false_color.png')
    plt.savefig(rgb_path, bbox_inches='tight', dpi=300)
    plt.close()
    # 再保存一个纯色没背景的HSI
    rgb_path2 = os.path.join(output_dir, f'{dataset_name}_false_color2.png')
    mpimg.imsave(rgb_path2, rgb_img)
    print(f'Saved false color image to: {rgb_path}')

    # 2. 保存GT标签图像
    plt.figure()
    gt_img, legend_elements = draw_gt(gt, imageID, labels)
    gt_path = os.path.join(output_dir, f'{dataset_name}_ground_truth.png')
    plt.gca().set_position([0, 0, 1, 1])  # 把图像填满整个画布
    plt.savefig(gt_path, bbox_inches='tight', dpi=300, pad_inches=0, transparent=True)
    plt.close()
    # 再保存一个纯色没背景的GT
    gt_path2 = os.path.join(output_dir, f'{dataset_name}_ground_truth2.png')
    mpimg.imsave(gt_path2, gt_img)

    # --- 单独保存图例 ---
    fig, ax = plt.subplots(figsize=(4, 6))  # 可以调大小
    ax.axis('off')  # 不要坐标轴

    # 只画图例，不画图像
    legend = ax.legend(handles=legend_elements,
                       loc='center',
                       frameon=False,  # 去掉图例边框
                       title="Classes")

    legend_path = os.path.join(output_dir, "legend_only.png")
    fig.savefig(legend_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

    print(f'Saved ground truth image to: {gt_path}')

    return rgb_path, gt_path


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Visualize HSI dataset')
    parser.add_argument('--dataset_name', type=str, default='sa',
                        help='Dataset name (e.g., pu, sa, ip)')
    parser.add_argument('--dataset_dir', type=str, default='./datasets',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for images')
    args = parser.parse_args()

    # 验证数据集名称
    if args.dataset_name not in DATASET_TO_ID:
        print(f"警告: 数据集 '{args.dataset_name}' 没有预定义的imageID映射")
        print("可用的数据集: ", list(DATASET_TO_ID.keys()))

    # 创建可视化图像
    rgb_path, gt_path = save_visualizations(
        args.dataset_name,
        args.dataset_dir,
        args.output_dir
    )

    print("\n可视化完成!")