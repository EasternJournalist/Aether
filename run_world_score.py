import json
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import argparse

from tqdm import tqdm
import numpy as np
from aether.utils.postprocess_utils import camera_pose_to_raymap

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=None)
    parser.add_argument('--rank', type=int, default=None)
    args = parser.parse_args()

    world_size = args.world_size
    rank = args.rank

    metainfo_root = Path('/mnt/blob/data_v4/HistoryWarp_long_v2/')
    metainfo_relpaths = Path(metainfo_root, '.metainfo_list.txt').read_text().strip().splitlines()

    for metainfo_relpath in tqdm(metainfo_relpaths[rank::world_size]):
        metainfo_abspath = Path(metainfo_root, metainfo_relpath)
        with open(metainfo_abspath, 'r') as f:
            metainfo = json.load(f)
        extrinsics = np.array(metainfo['camera_extrinsics'])
        H, W = 480, 720
        frame_indices = np.linspace(0, len(extrinsics) - 1, 41, dtype=int)
        extrinsics = extrinsics[frame_indices]
        extrinsics = extrinsics @ np.linalg.inv(extrinsics[0])
        raymap = camera_pose_to_raymap(
            camera_pose=np.linalg.inv(extrinsics),
            intrinsic=np.array(metainfo['camera_intrinsics'], dtype=np.float32)[frame_indices] * np.array([W, H, 1], dtype=np.float32)[:, None],
        )

        with TemporaryDirectory() as tmpdirname:
            raymap_path = Path(tmpdirname, 'raymap.npy')
            np.save(raymap_path, raymap)

            image_path = metainfo_abspath.with_name('input_image.png')
            for seed in [0, 1]:
                output_path = Path('/mnt/blob/workspace/aether', Path(metainfo_relpath).parent, f'seed_{seed}')
                output_path.mkdir(parents=True, exist_ok=True)
                cmd = f'python scripts/demo.py --task prediction --image {image_path} --raymap_action {raymap_path} --output_dir {output_path} --num_frames 41'
                print("Executing command:", cmd)
                os.system(cmd)
