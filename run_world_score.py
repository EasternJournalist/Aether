import json
from pathlib import Path
from tempfile import TemporaryDirectory
import os

from tqdm import tqdm
import numpy as np
from aether.utils.postprocess_utils import camera_pose_to_raymap

if __name__ == "__main__":
    metainfo_root = Path('/mnt/blob/data_v4/HistoryWarp_long_v2/')
    metainfo_paths = list(tqdm(Path('/mnt/blob/data_v4/HistoryWarp_long_v2/worldscore_output/static/photorealistic').rglob('metainfo.json'), desc='metainfo_paths'))

    for metainfo_path in tqdm(metainfo_paths):
        with open(metainfo_path, 'r') as f:
            metainfo = json.load(f)

        H, W = 480, 720
        frame_indices = np.linspace(0, len(metainfo['camera_extrinsics']) - 1, 41, dtype=int)
        raymap = camera_pose_to_raymap(
            camera_pose=np.linalg.inv(metainfo['camera_extrinsics'])[frame_indices],
            intrinsic=np.array(metainfo['camera_intrinsics'], dtype=np.float32)[frame_indices] * np.array([W, H, 1], dtype=np.float32),
        )

        with TemporaryDirectory() as tmpdirname:
            raymap_path = Path(tmpdirname, 'raymap.npy')
            np.save(raymap_path, raymap)

            image_path = metainfo_path.with_name('input_image.png')
            output_path = Path('/mnt/blob/workspace/aether', metainfo_path.relative_to(metainfo_root))
            os.system(f'python scripts/demo.py --task prediction --image {image_path} --raymap_action {raymap_path} --output_dir {output_path} --num_frames 41')
