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

        matrixes = motion.animate_bone(
            list(range(motion.bones.max_fno)),
            trace_model,
            bone_names=[
                "左つま先親",
                "左つま先子",
                "左つま先",
                "左かかと",
                "左足首",
                "右つま先親",
                "右つま先子",
                "右つま先",
                "右かかと",
                "右足首",
            ],
            is_calc_ik=True,
            out_fno_log=True,
            description="事前計算",
        )

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

            for direction in ["左", "右"]:
                leg_ik_name = f"{direction}足ＩＫ"
                heel_name = f"{direction}かかと"
                toe_name = f"{direction}つま先"
                big_toe_name = f"{direction}つま先親"
                small_toe_name = f"{direction}つま先子"

                # Yが0の場合、足首の向きを調整して接地させる
                if motion.bones[leg_ik_name][fno].position.y == 0:
                    heel_pos = matrixes[heel_name, fno].position
                    toe_pos = matrixes[toe_name, fno].position
                    toe_ground_pos = MVector3D(toe_pos.x, heel_pos.y, toe_pos.z)

                    toe_local_pos = (toe_pos - heel_pos).normalized()
                    toe_ground_local_pos = (toe_ground_pos - heel_pos).normalized()

                    # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
                    # 回転角
                    ankle_y_rotation_rad: float = np.arccos(
                        np.clip(
                            toe_local_pos.dot(toe_ground_local_pos),
                            -1,
                            1,
                        )
                    )

                    # 回転軸
                    ankle_y_rotation_axis: MVector3D = toe_local_pos.cross(
                        toe_ground_local_pos
                    ).normalized()

                    # Y方向の角度
                    ankle_y_qq = MQuaternion.from_axis_angles(ankle_y_rotation_axis, ankle_y_rotation_rad)

                    # -----------------------
                    big_toe_pos = matrixes[big_toe_name, fno].position
                    small_toe_pos = matrixes[small_toe_name, fno].position
                    small_ground_pos = MVector3D(small_toe_pos.x, big_toe_pos.y, small_toe_pos.z)

                    small_toe_local_pos = (small_toe_pos - big_toe_pos).normalized()
                    small_toe_ground_local_pos = (small_ground_pos - big_toe_pos).normalized()

                    # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
                    # 回転角
                    ankle_x_rotation_rad: float = np.arccos(
                        np.clip(
                            small_toe_local_pos.dot(small_toe_ground_local_pos),
                            -1,
                            1,
                        )
                    )

                    # 回転軸
                    ankle_x_rotation_axis: MVector3D = small_toe_local_pos.cross(
                        small_toe_ground_local_pos
                    ).normalized()

                    # X方向の角度
                    ankle_x_qq = MQuaternion.from_axis_angles(ankle_x_rotation_axis, ankle_x_rotation_rad)

                    # 足首の向きを調整する角度
                    motion.bones[leg_ik_name][fno].rotation = ankle_x_qq * ankle_y_qq * motion.bones[leg_ik_name][fno].rotation

        VmdWriter(
            motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_ground_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()

        VmdWriter(
            ground_motion,
            os.path.join(args.target_dir, f"output_poses_mp_rot_ik_arm_ik_ground_only_{person_id:02d}.vmd"),
            "4D-Humans",
        ).save()
