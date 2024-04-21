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
        "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_arm_ik2_model.pmx"
        # "/mnt/d/MMD/MikuMikuDance_v926x64/UserFile/Model/_あにまさ式ミク準標準見せパン/初音ミクVer2 準標準 見せパン 3_trace_ik.pmx"
    )

    for person_id in range(10):
        motion_path = os.path.join(
            args.target_dir, f"output_poses_mp_rot_ik_leg_{person_id:02d}.vmd"
        )
        if not os.path.exists(motion_path):
            break

        ik_motion = VmdReader().read_by_filepath(motion_path)

        number_motion = VmdMotion()

        matrixes = ik_motion.animate_bone(
            list(range(ik_motion.bones.max_fno)),
            trace_model,
            bone_names=["左手首", "右手首"],
            is_calc_ik=False,
            out_fno_log=True,
            description="事前計算",
        )

        # 腕IKの設定
        for direction in ["左", "右"]:
            arm_ik_name = f"{direction}腕IK"

            arm_name = f"{direction}腕"
            arm_twist_name = f"{direction}腕捩"
            elbow_name = f"{direction}ひじ"
            elbow_y_perpendicular_name = f"{direction}ひじY垂線"
            elbow_z_perpendicular_name = f"{direction}ひじZ垂線"
            wrist_name = f"{direction}手首"

            del ik_motion.bones[arm_name]
            del ik_motion.bones[elbow_name]

            for fno in tqdm(range(ik_motion.bones.max_fno), desc=f"{direction}腕"):
                # 腕回転から捩りを除去した角度をローカルY軸に沿って回転
                arm_x_qq, _, _, arm_yz_qq = matrixes[
                    arm_name, fno
                ].frame_fk_rotation.separate_by_axis(
                    trace_model.bones[arm_name].corrected_local_x_vector
                )
                arm_bf = VmdBoneFrame(name=arm_name, index=fno, register=True)
                arm_bf.rotation = arm_yz_qq
                ik_motion.append_bone_frame(arm_bf)

                # ひじ回転から捩りを除去した角度をローカルY軸に沿って回転
                elbow_x_qq, _, _, elbow_yz_qq = matrixes[
                    elbow_name, fno
                ].frame_fk_rotation.separate_by_axis(
                    trace_model.bones[elbow_name].corrected_local_x_vector
                )
                elbow_bf = VmdBoneFrame(name=elbow_name, index=fno, register=True)
                elbow_bf.rotation = MQuaternion.from_axis_angles(
                    trace_model.bones[elbow_name].corrected_local_y_vector,
                    elbow_yz_qq.to_radian(),
                )
                ik_motion.append_bone_frame(elbow_bf)

                # 腕IKの終着地点として手首の位置を取得
                arm_ik_bf = VmdBoneFrame(name=arm_ik_name, index=fno, register=True)
                arm_ik_bf.position = matrixes[wrist_name, fno].position
                ik_motion.append_bone_frame(arm_ik_bf)

                elbow_pos_bf = VmdBoneFrame(
                    name=("0" if direction == "左" else "10"), index=fno, register=True
                )
                elbow_pos_bf.position = matrixes[elbow_name, fno].position
                number_motion.append_bone_frame(elbow_pos_bf)

                wrist_pos_bf = VmdBoneFrame(
                    name=("1" if direction == "左" else "11"), index=fno, register=True
                )
                wrist_pos_bf.position = matrixes[wrist_name, fno].position
                number_motion.append_bone_frame(wrist_pos_bf)

        VmdWriter(
            number_motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_pos_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()

        VmdWriter(
            ik_motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_pre_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()

        ik_matrixes = ik_motion.animate_bone(
            list(range(ik_motion.bones.max_fno)),
            trace_model,
            bone_names=["左手首", "右手首"],
            is_calc_ik=True,
            out_fno_log=True,
            description="IK計算",
        )

        # 腕捩FKの再設定
        for direction in ["左", "右"]:
            arm_ik_name = f"{direction}腕IK"

            arm_name = f"{direction}腕"
            arm_twist_name = f"{direction}腕捩"
            elbow_name = f"{direction}ひじ"
            elbow_y_perpendicular_name = f"{direction}ひじY垂線"
            elbow_z_perpendicular_name = f"{direction}ひじZ垂線"
            wrist_name = f"{direction}手首"

            for fno in tqdm(range(ik_motion.bones.max_fno), desc=f"{direction}腕"):
                arm_twist_bf = VmdBoneFrame(name=arm_twist_name, index=fno, register=True)
                arm_twist_bf.rotation = ik_matrixes[arm_twist_name, fno].frame_fk_rotation
                ik_motion.append_bone_frame(arm_twist_bf)

            del ik_motion.bones[arm_ik_name]

        VmdWriter(
            ik_motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()
