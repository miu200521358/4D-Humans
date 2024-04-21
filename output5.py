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
    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_model.pmx"
        # "/mnt/d/MMD/MikuMikuDance_v926x64/UserFile/Model/_あにまさ式ミク準標準見せパン/初音ミクVer2 準標準 見せパン 3_trace.pmx"
    )

    for person_id in range(10):
        motion_path = os.path.join(
            args.target_dir, f"output_poses_mp_rot_{person_id:02d}.vmd"
        )
        if not os.path.exists(motion_path):
            break

        poses_rot_motion = VmdReader().read_by_filepath(motion_path)

        ik_motion = poses_rot_motion.copy()
        ik_set_motion = poses_rot_motion.copy()
        matrixes = poses_rot_motion.animate_bone(
            list(range(poses_rot_motion.bones.max_fno)),
            trace_model,
            bone_names=["右つま先", "左つま先"],
            is_calc_ik=False,
            out_fno_log=True,
            description="事前計算",
        )

        # 足IKの設定
        for direction in ["左", "右"]:
            leg_ik_parent_name = f"{direction}足IK親"
            leg_name = f"{direction}足"
            knee_name = f"{direction}ひざ"
            ankle_name = f"{direction}足首"
            toe_name = f"{direction}つま先"
            leg_ik_name = f"{direction}足ＩＫ"

            arm_name = f"{direction}腕"
            elbow_name = f"{direction}ひじ"
            wrist_name = f"{direction}手首"

            del ik_motion.bones[leg_name]
            del ik_motion.bones[knee_name]
            del ik_motion.bones[ankle_name]

            for fno in tqdm(
                range(poses_rot_motion.bones.max_fno), desc=f"{direction}足"
            ):
                leg_ik_bf = VmdBoneFrame(name=leg_ik_name, index=fno, register=True)
                leg_ik_bf.position = (
                    matrixes[ankle_name, fno].position
                    - trace_model.bones[ankle_name].position
                )
                leg_ik_bf.rotation = matrixes[
                    ankle_name, fno
                ].local_matrix.to_quaternion()
                ik_motion.append_bone_frame(leg_ik_bf)

                ik_set_motion.append_bone_frame(leg_ik_bf)

                knee_x_qq, _, _, knee_yz_qq = matrixes[
                    knee_name, fno
                ].frame_fk_rotation.separate_by_axis(
                    trace_model.bones[knee_name].corrected_local_x_vector
                )

                # ひざを角度制限に合わせて変換する
                knee_bf = VmdBoneFrame(name=knee_name, index=fno, register=True)
                knee_bf.rotation = MQuaternion.from_axis_angles(
                    MVector3D(1, 0, 0), -knee_yz_qq.to_radian()
                )
                ik_motion.append_bone_frame(knee_bf)

                leg_bf = ik_motion.bones[leg_name][fno]
                leg_bf.rotation = knee_x_qq * leg_bf.rotation
                ik_motion.append_bone_frame(leg_bf)

        VmdWriter(
            ik_set_motion,
            os.path.join(
                args.target_dir, f"output_poses_mp_rot_ik_leg_set_{person_id:02d}.vmd"
            ),
            "4D-Humans",
        ).save()

        ik_matrixes = poses_rot_motion.animate_bone(
            list(range(poses_rot_motion.bones.max_fno)),
            trace_model,
            bone_names=["右つま先", "左つま先"],
            is_calc_ik=True,
            out_fno_log=True,
            description="IK計算",
        )

        # 足FKの再設定
        for direction in ["左", "右"]:
            leg_ik_parent_name = f"{direction}足IK親"
            leg_name = f"{direction}足"
            knee_name = f"{direction}ひざ"
            ankle_name = f"{direction}足首"
            toe_name = f"{direction}つま先"
            leg_ik_name = f"{direction}足ＩＫ"

            arm_name = f"{direction}腕"
            elbow_name = f"{direction}ひじ"
            wrist_name = f"{direction}手首"

            del ik_motion.bones[leg_name]
            del ik_motion.bones[knee_name]
            del ik_motion.bones[ankle_name]

            for fno in tqdm(
                range(poses_rot_motion.bones.max_fno), desc=f"{direction}足"
            ):
                leg_bf = VmdBoneFrame(name=leg_name, index=fno, register=True)
                leg_bf.rotation = ik_matrixes[leg_name, fno].frame_fk_rotation
                ik_motion.append_bone_frame(leg_bf)

                knee_bf = VmdBoneFrame(name=knee_name, index=fno, register=True)
                knee_bf.rotation = ik_matrixes[knee_name, fno].frame_fk_rotation
                ik_motion.append_bone_frame(knee_bf)

                ankle_bf = VmdBoneFrame(name=ankle_name, index=fno, register=True)
                ankle_bf.rotation = ik_matrixes[ankle_name, fno].frame_fk_rotation
                ik_motion.append_bone_frame(ankle_bf)

        VmdWriter(
            ik_motion,
            os.path.join(
                args.target_dir, f"output_poses_mp_rot_ik_leg_{person_id:02d}.vmd"
            ),
            "4D-Humans",
        ).save()
