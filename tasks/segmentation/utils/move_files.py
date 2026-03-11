# 将A文件夹复制到B文件夹，并排除指定名称文件
from datetime import datetime
import os
import shutil


def move_files(src_dir, dst_dir, exclude_names):
    """
    将源目录中的文件和子目录结构完整复制到目标目录，排除指定名称的文件。
    保持源目录的原有结构，原文件将保留在源目录中，不会被删除。

    参数:
        src_dir (str): 源目录路径，从中复制文件和目录
        dst_dir (str): 目标目录路径，复制文件和目录到此目录
        exclude_names (list or set): 要排除的文件名列表或集合，这些文件不会被复制

    返回:
        None
    """

    for root, dirs, files in traverse_directory(src_dir, exclude_names):
        relpath = os.path.relpath(root, src_dir)
        if os.path.basename(relpath) in exclude_names:
            continue
        # print(f"当前目录: {root}")
        # print(f"当前相对目录: {relpath}")
        for dir_name in dirs:
            if dir_name not in exclude_names:
                # print(f"子目录: {dir_name}")
                dst_path = os.path.join(dst_dir, relpath, dir_name)
                # 创建目标目录（如果不存在）
                if not os.path.exists(dst_path):
                    os.makedirs(dst_path)

        # 复制文件
        for file in files:
            if file not in exclude_names:
                # print(f"文件: {file}")
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, relpath, file)
                shutil.copy2(src_path, dst_path)


def traverse_directory(root_dir, exclude_names=None):
    """
    遍历一个文件夹内所有子文件夹和文件，包括子文件夹内的子文件夹，并可排除指定的文件或文件夹
    
    参数:
        root_dir (str): 需要遍历的根目录路径
        exclude_names (list or set, optional): 要排除的文件名或文件夹名列表，默认为None
        
    返回:
        generator: 返回一个生成器，每次产生一个(目录路径, 子目录列表, 文件列表)的元组
    """
    if exclude_names is None:
        exclude_names = set()
    elif isinstance(exclude_names, (list, tuple)):
        exclude_names = set(exclude_names)

    for root, dirs, files in os.walk(root_dir):
        # 过滤掉被排除的文件
        filtered_files = [f for f in files if f not in exclude_names]

        # 从dirs中移除被排除的目录，这样os.walk就不会遍历这些目录
        dirs[:] = [d for d in dirs if d not in exclude_names]

        yield root, dirs, filtered_files


if __name__ == '__main__':
    # src_dict = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation"
    # # dst_dir = r'D:\data\segmentation\train\images_copy'
    # exclude_names = ['exclude_file1.txt', 'exclude_file2.txt']
    # for root, dirs, files in traverse_directory(src_dict):
    #     print(f"当前目录: {root}")
    #     for dir_name in dirs:
    #         print(f"子目录: {dir_name}")
    #     for file_name in files:
    #         print(f"文件: {file_name}")

    # 删除某文件夹
    # shutil.rmtree(src_dict + "/logs")

    # a_dir = '/home/yyyjvm/SS-projects/dinov3/tasks/segmentation/model'
    # root_dir = '/home/yyyjvm/SS-projects/dinov3/tasks/segmentation'
    # print(os.path.relpath(a_dir, root_dir))

    # date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # src_dict = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation"
    # dst_dict = f"{src_dict}/logs/{date_time}"
    # move_files(src_dict, dst_dict, ['logs', '__pycache__', '.pyc'])

    print(
        os.path.basename(
            '/home/yyyjvm/SS-projects/dinov3/tasks/segmentation/model'))
