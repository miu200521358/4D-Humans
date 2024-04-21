import argparse
from glob import glob
import json
import os
from pykalman import UnscentedKalmanFilter

import numpy as np
from tqdm import tqdm
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame
from mlib.vmd.vmd_writer import VmdWriter
from mlib.vmd.vmd_reader import VmdReader
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_collection import PmxModel
from mlib.core.math import MVector3D, MQuaternion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMR2 demo code")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="demo_out",
        help="Output folder to save rendered results",
    )

    args = parser.parse_args()

    pmx_reader = PmxReader()
    # trace_model = pmx_reader.read_by_filepath(
    #     "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_model.pmx"
    # )

    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_model.pmx"
        # "/mnt/d/MMD/MikuMikuDance_v926x64/UserFile/Model/_あにまさ式ミク準標準見せパン/初音ミクVer2 準標準 見せパン 3_trace.pmx"
    )

    for person_id in range(10):
        motion_path = os.path.join(
            args.target_dir, f"output_poses_mp_rot_ik_arm_ik_{person_id:02d}.vmd"
        )
        if not os.path.exists(motion_path):
            break

        motion = VmdReader().read_by_filepath(motion_path)
        ground_motion = motion.copy()

        leg_ik_ys = []

        for fno in tqdm(range(motion.bones.max_fno), desc="足ＩＫ-Y準備"):
            leg_ik_ys.append(
                min(
                    motion.bones["左足ＩＫ"][fno].position.y,
                    motion.bones["右足ＩＫ"][fno].position.y,
                )
            )

        # 9割は接地していると仮定して、足IKのY座標を補正
        leg_ik_ys = np.array(leg_ik_ys)
        ground_y = np.percentile(leg_ik_ys, 90)

        for fno in tqdm(range(motion.bones.max_fno), desc="足ＩＫ-Y補正"):
            motion.bones["左足ＩＫ"][fno].position.y = max(
                motion.bones["左足ＩＫ"][fno].position.y - ground_y, 0
            )
            motion.bones["右足ＩＫ"][fno].position.y = max(
                motion.bones["右足ＩＫ"][fno].position.y - ground_y, 0
            )
            motion.bones["センター"][fno].position.y -= ground_y

            ground_motion.bones["左足ＩＫ"][fno].position.y = max(
                ground_motion.bones["左足ＩＫ"][fno].position.y - ground_y, 0
            )
            ground_motion.bones["右足ＩＫ"][fno].position.y = max(
                ground_motion.bones["右足ＩＫ"][fno].position.y - ground_y, 0
            )
            ground_motion.bones["センター"][fno].position.y -= ground_y

        VmdWriter(
            ground_motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_ground_only_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()
