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

    twist_motion = VmdReader().read_by_filepath(
        os.path.join(args.target_dir, "output_poses_mp_rot_ik_leg.vmd")
    )
    twist_number_motion = VmdMotion()

    pmx_reader = PmxReader()
    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_twist3_model.pmx"
    )

    matrixes = twist_motion.animate_bone(
        list(range(twist_motion.bones.max_fno)),
        trace_model,
        bone_names=[
            "左手首",
            "右手首",
            "左ひじZ垂線",
            "右ひじZ垂線",
            "左ひじY垂線",
            "右ひじY垂線",
        ],
        is_calc_ik=False,
        out_fno_log=True,
        description="事前計算",
    )

    # 腕の設定
    for direction in ["左", "右"]:
        arm_name = f"{direction}腕"
        arm_twist_name = f"{direction}腕捩"
        elbow_name = f"{direction}ひじ"
        wrist_name = f"{direction}手首"
        elbow_y_perpendicular_name = f"{direction}ひじY垂線"
        elbow_z_perpendicular_name = f"{direction}ひじZ垂線"

        for fno in tqdm(range(twist_motion.bones.max_fno), desc=f"{direction}腕"):
            # 腕回転から捩りを除去した角度をローカルY軸に沿って回転
            arm_x_qq, _, _, arm_yz_qq = matrixes[
                arm_name, fno
            ].frame_fk_rotation.separate_by_axis(
                trace_model.bones[arm_name].corrected_local_x_vector
            )
            arm_bf = VmdBoneFrame(name=arm_name, index=fno, register=True)
            arm_bf.rotation = arm_yz_qq
            twist_motion.append_bone_frame(arm_bf)

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
            twist_motion.append_bone_frame(elbow_bf)

            # twist_bf = VmdBoneFrame(name=arm_twist_name, index=fno, register=True)
            # twist_bf.rotation = MQuaternion.from_axis_angles(
            #     trace_model.bones[arm_twist_name].corrected_fixed_axis,
            #     elbow_x_qq.to_radian(),
            # )
            # twist_motion.append_bone_frame(twist_bf)

            wrist_pos = matrixes[wrist_name, fno].position

            # ひじのZ軸回転があると動く
            elbow_y_perpendicular_pos = matrixes[
                elbow_y_perpendicular_name, fno
            ].position

            # ひじのY軸回転があると動く
            elbow_z_perpendicular_pos = matrixes[
                elbow_z_perpendicular_name, fno
            ].position

            # 腕のグロバール行列
            arm_global_matrix = matrixes[arm_name, fno].global_matrix.copy()
            # ひじの角度が無い状態でひじまでの相対位置を追加する
            arm_global_matrix.translate(
                trace_model.bones[elbow_name].position
                - trace_model.bones[arm_name].position
            )
            # ひじのY軸のみの回転
            arm_global_matrix.rotate(elbow_bf.rotation)

            # ひじのZ軸回転が無い場合のひじY垂線の位置
            elbow_y_perpendicular_pos_no_twist = arm_global_matrix * (
                trace_model.bones[elbow_y_perpendicular_name].position
                - trace_model.bones[elbow_name].position
            )

            # ひじのY軸回転が無い場合のひじZ垂線の位置
            elbow_z_perpendicular_pos_no_twist = arm_global_matrix * (
                trace_model.bones[elbow_z_perpendicular_name].position
                - trace_model.bones[elbow_name].position
            )

            # ひじY回転のローカル位置
            elbow_z_perpendicular_local_pos = (
                elbow_z_perpendicular_pos - matrixes[elbow_name, fno].position
            )

            # ひじZ回転のローカル位置
            elbow_y_perpendicular_local_pos = (
                elbow_y_perpendicular_pos - matrixes[elbow_name, fno].position
            )

            # ひじZ回転なしのローカル位置
            elbow_y_perpendicular_no_twist_local_pos = (
                elbow_y_perpendicular_pos_no_twist - matrixes[elbow_name, fno].position
            )

            # ひじY回転なしのローカル位置
            elbow_z_perpendicular_no_twist_local_pos = (
                elbow_z_perpendicular_pos_no_twist - matrixes[elbow_name, fno].position
            )

            from_pos = elbow_z_perpendicular_no_twist_local_pos
            to_pos = elbow_y_perpendicular_local_pos

            # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
            # 回転角
            rotation_rad: float = np.arccos(
                np.clip(
                    from_pos.dot(to_pos),
                    -1,
                    1,
                )
            )

            # 回転軸
            rotation_axis: MVector3D = (
                from_pos.normalized().cross(to_pos.normalized()).normalized()
            )

            plus_qq = MQuaternion.from_axis_angles(
                trace_model.bones[arm_twist_name].corrected_fixed_axis,
                -rotation_rad * np.sign(trace_model.bones[arm_twist_name].corrected_fixed_axis.x),
            )
            minus_qq = MQuaternion.from_axis_angles(
                trace_model.bones[arm_twist_name].corrected_fixed_axis, -rotation_rad
            )

            twist_lengths = []
            for t, qq in enumerate([plus_qq, minus_qq]):
                # 腕のグロバール行列
                arm_global_matrix = matrixes[arm_name, fno].global_matrix.copy()
                # 腕捩
                arm_global_matrix.translate(
                    trace_model.bones[arm_twist_name].position
                    - trace_model.bones[arm_name].position
                )
                arm_global_matrix.rotate(qq)
                # ひじ
                arm_global_matrix.translate(
                    trace_model.bones[elbow_name].position
                    - trace_model.bones[arm_twist_name].position
                )
                arm_global_matrix.rotate(elbow_bf.rotation)
                # 手首
                arm_global_matrix.translate(
                    trace_model.bones[wrist_name].position
                    - trace_model.bones[elbow_name].position
                )
                twist_pos = arm_global_matrix.to_position()
                twist_lengths.append((twist_pos - wrist_pos).length())

                if direction == "左":
                    if t == 0:
                        twist_pos_name = "20"
                    else:
                        twist_pos_name = "21"
                else:
                    if t == 0:
                        twist_pos_name = "22"
                    else:
                        twist_pos_name = "23"
                twist_pos_bf = VmdBoneFrame(
                    name=twist_pos_name, index=fno, register=True
                )
                twist_pos_bf.position = twist_pos
                twist_pos_bf.rotation = qq
                twist_number_motion.append_bone_frame(twist_pos_bf)

            # 差が小さい方を採用する
            twist_qq = [plus_qq, minus_qq][int(np.argmin(twist_lengths))]
            print(
                f"[{direction}][{fno}] {int(np.argmin(twist_lengths))}({twist_lengths})"
            )

            # if test_qq.xyz.dot(rotation_axis) < 0:
            #     rotation_rad = -rotation_rad

            # twist_qq = MQuaternion.from_axis_angles(
            #     trace_model.bones[arm_twist_name].corrected_fixed_axis, rotation_rad
            # )
            arm_twist_bf = VmdBoneFrame(name=arm_twist_name, index=fno, register=True)
            arm_twist_bf.rotation = twist_qq
            twist_motion.append_bone_frame(arm_twist_bf)

            elbow_pos_bf = VmdBoneFrame(
                name=("0" if direction == "左" else "10"), index=fno, register=True
            )
            elbow_pos_bf.position = matrixes[elbow_name, fno].position
            twist_number_motion.append_bone_frame(elbow_pos_bf)

            elbow_y_perpendicular_pos_bf = VmdBoneFrame(
                name=("1" if direction == "左" else "11"), index=fno, register=True
            )
            elbow_y_perpendicular_pos_bf.position = elbow_y_perpendicular_pos
            twist_number_motion.append_bone_frame(elbow_y_perpendicular_pos_bf)

            elbow_y_perpendicular_pos_bf = VmdBoneFrame(
                name=("2" if direction == "左" else "12"), index=fno, register=True
            )
            elbow_y_perpendicular_pos_bf.position = elbow_y_perpendicular_pos_no_twist
            twist_number_motion.append_bone_frame(elbow_y_perpendicular_pos_bf)

            elbow_z_perpendicular_pos_bf = VmdBoneFrame(
                name=("3" if direction == "左" else "13"), index=fno, register=True
            )
            elbow_z_perpendicular_pos_bf.position = elbow_z_perpendicular_pos
            twist_number_motion.append_bone_frame(elbow_z_perpendicular_pos_bf)

            elbow_z_perpendicular_pos_bf = VmdBoneFrame(
                name=("4" if direction == "左" else "14"), index=fno, register=True
            )
            elbow_z_perpendicular_pos_bf.position = elbow_z_perpendicular_pos_no_twist
            twist_number_motion.append_bone_frame(elbow_z_perpendicular_pos_bf)

    # twist_matrixes = twist_motion.animate_bone(
    #     list(range(twist_motion.bones.max_fno)),
    #     trace_model,
    #     bone_names=[
    #         "左手首",
    #         "右手首",
    #         "左ひじZ垂線",
    #         "右ひじZ垂線",
    #         "左ひじY垂線",
    #         "右ひじY垂線",
    #     ],
    #     is_calc_ik=False,
    #     out_fno_log=True,
    #     description="捩り計算",
    # )

    # # 腕の設定
    # for direction in ["左", "右"]:
    #     arm_name = f"{direction}腕"
    #     arm_twist_name = f"{direction}腕捩"
    #     elbow_name = f"{direction}ひじ"
    #     elbow_y_perpendicular_name = f"{direction}ひじY垂線"
    #     elbow_z_perpendicular_name = f"{direction}ひじZ垂線"
    #     wrist_twist_name = f"{direction}手捩"
    #     wrist_name = f"{direction}手首"

    #     for fno in tqdm(range(twist_motion.bones.max_fno), desc=f"{direction}腕"):
    #         # 元々のひじY垂線の位置
    #         original_elbow_y_perpendicular_pos = matrixes[
    #             elbow_y_perpendicular_name, fno
    #         ].position

    #         # 元々のひじZ垂線の位置
    #         original_elbow_z_perpendicular_pos = matrixes[
    #             elbow_z_perpendicular_name, fno
    #         ].position

    #         # ひじのY軸制限後のひじY垂線の位置
    #         twist_elbow_y_perpendicular_pos = twist_matrixes[
    #             elbow_y_perpendicular_name, fno
    #         ].position

    #         # ひじのY軸制限後のひじZ垂線の位置
    #         twist_elbow_z_perpendicular_pos = twist_matrixes[
    #             elbow_z_perpendicular_name, fno
    #         ].position

    #         # ひじY垂線のローカル位置
    #         original_elbow_y_perpendicular_local_pos = (
    #             original_elbow_y_perpendicular_pos - matrixes[elbow_name, fno].position
    #         ).normalized()
    #         twist_elbow_y_perpendicular_local_pos = (
    #             twist_elbow_y_perpendicular_pos - matrixes[elbow_name, fno].position
    #         ).normalized()

    #         # ひじZ垂線のローカル位置
    #         original_elbow_z_perpendicular_local_pos = (
    #             original_elbow_z_perpendicular_pos - matrixes[elbow_name, fno].position
    #         ).normalized()
    #         twist_elbow_z_perpendicular_local_pos = (
    #             twist_elbow_z_perpendicular_pos - matrixes[elbow_name, fno].position
    #         ).normalized()

    #         # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
    #         # Y軸回転角
    #         y_rotation_rad: float = np.arccos(
    #             np.clip(
    #                 twist_elbow_y_perpendicular_local_pos.dot(
    #                     original_elbow_y_perpendicular_local_pos
    #                 ),
    #                 -1,
    #                 1,
    #             )
    #         )

    #         # Y軸回転軸
    #         y_rotation_axis: MVector3D = (
    #             twist_elbow_y_perpendicular_local_pos.normalized()
    #             .cross(original_elbow_y_perpendicular_local_pos.normalized())
    #             .normalized()
    #         )

    #         y_twist_qq = MQuaternion.from_axis_angles(y_rotation_axis, y_rotation_rad)

    #         # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
    #         # Z軸回転角
    #         z_rotation_rad: float = np.arccos(
    #             np.clip(
    #                 twist_elbow_z_perpendicular_local_pos.dot(
    #                     original_elbow_z_perpendicular_local_pos
    #                 ),
    #                 -1,
    #                 1,
    #             )
    #         )

    #         # Z軸回転軸
    #         z_rotation_axis: MVector3D = (
    #             twist_elbow_z_perpendicular_local_pos.normalized()
    #             .cross(original_elbow_z_perpendicular_local_pos.normalized())
    #             .normalized()
    #         )

    #         # Z軸の変化を検出して、Y軸の回転を除去
    #         z_twist_qq = MQuaternion.from_axis_angles(z_rotation_axis, z_rotation_rad)

    #         twist_qq = y_twist_qq * z_twist_qq.inverse()
    #         twist_rad = twist_qq.to_radian()
    #         if (
    #             twist_qq.xyz.dot(trace_model.bones[arm_twist_name].corrected_fixed_axis)
    #             < 0
    #         ):
    #             twist_rad = -twist_rad

    #         # 腕捩の回転量
    #         arm_twist_bf = VmdBoneFrame(name=arm_twist_name, index=fno, register=True)
    #         arm_twist_bf.rotation = MQuaternion.from_axis_angles(
    #             trace_model.bones[arm_twist_name].corrected_fixed_axis,
    #             twist_rad,
    #         )
    #         twist_motion.append_bone_frame(arm_twist_bf)

    VmdWriter(
        twist_motion,
        os.path.join(args.target_dir, "output_poses_mp_rot_ik_leg_twist_arm.vmd"),
        "4D-Humans",
    ).save()

    VmdWriter(
        twist_number_motion,
        os.path.join(
            args.target_dir, "output_poses_mp_rot_ik_leg_twist_arm_number.vmd"
        ),
        "4D-Humans",
    ).save()
