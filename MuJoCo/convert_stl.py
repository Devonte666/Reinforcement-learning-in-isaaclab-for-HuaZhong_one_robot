# convert_stl.py
from stl import mesh
import os

mesh_dir = "/home/user/Devonte_file/Reinforcement-learning-in-isaaclab-for-HuaZhong_one_robot/Huazhong1/meshes"

for filename in os.listdir(mesh_dir):
    if filename.endswith(".STL"):
        filepath = os.path.join(mesh_dir, filename)
        try:
            m = mesh.Mesh.from_file(filepath)
            # 直接保存，默认二进制
            m.save(filepath)
            print(f"Converted: {filename}")
        except Exception as e:
            print(f"Failed: {filename} - {e}")