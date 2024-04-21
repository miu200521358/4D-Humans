

import argparse
import os
import joblib
from output import JOINT_NAMES, PMX_CONNECTIONS, VMD_CONNECTIONS, MIKU_CM
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame
from mlib.vmd.vmd_writer import VmdWriter
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_collection import PmxModel
from mlib.core.math import MVector3D, MQuaternion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--pkl_path', type=str)
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')

    args = parser.parse_args()

    image2 = joblib.load(args.pkl_path)

    poses_mov_motion = VmdMotion()
    poses_rot_motion = VmdMotion()

    pmx_reader = PmxReader()
    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-3/data/pmx/trace_model.pmx"
    )

    for i, (img_path, data) in enumerate(image2.items()):
        joints = data["3d_joints"]
        for jname, joint in zip(JOINT_NAMES, joints[0]):
            if jname not in PMX_CONNECTIONS:
                pose_bf = VmdBoneFrame(i, jname, register=True)
            else:
                pose_bf = VmdBoneFrame(i, PMX_CONNECTIONS[jname], register=True)
            pose_bf.position = MVector3D(joint[0], -joint[1], joint[2])
            pose_bf.position *= 10

            poses_mov_motion.append_bone_frame(pose_bf)

        pelvis2_bf = VmdBoneFrame(i, "下半身2", register=True)
        pelvis2_bf.position = (
            poses_mov_motion.bones["左ひざ"][i].position
            + poses_mov_motion.bones["右ひざ"][i].position
        ) / 2
        poses_mov_motion.append_bone_frame(pelvis2_bf)

        camera_bbox = data["camera_bbox"][0]
        pelvis_bf = poses_mov_motion.bones["下半身"][i]
        center_bf = VmdBoneFrame(i, "センター", register=True)
        center_bf.position = MVector3D(camera_bbox[0], -camera_bbox[1], camera_bbox[2]) / MIKU_CM
        poses_rot_motion.append_bone_frame(center_bf)

    VmdWriter(
        poses_mov_motion,
        os.path.join(args.out_folder, "output_poses_mov.vmd"),
        "4D-Humans",
    ).save()

    for target_bone_name, vmd_params in VMD_CONNECTIONS.items():
        direction_from_name = vmd_params["direction"][0]
        direction_to_name = vmd_params["direction"][1]
        up_from_name = vmd_params["up"][0]
        up_to_name = vmd_params["up"][1]
        cross_from_name = (
            vmd_params["cross"][0]
            if "cross" in vmd_params
            else vmd_params["direction"][0]
        )
        cross_to_name = (
            vmd_params["cross"][1]
            if "cross" in vmd_params
            else vmd_params["direction"][1]
        )
        cancel_names = vmd_params["cancel"]
        invert_qq = MQuaternion.from_euler_degrees(vmd_params["invert"]["before"])

        for mov_bf in poses_mov_motion.bones[direction_from_name]:
            if (
                mov_bf.index not in poses_mov_motion.bones[direction_from_name]
                or mov_bf.index not in poses_mov_motion.bones[direction_to_name]
            ):
                # キーがない場合、スルーする
                continue

            bone_direction = (
                trace_model.bones[direction_to_name].position
                - trace_model.bones[direction_from_name].position
            ).normalized()

            bone_up = (
                trace_model.bones[up_to_name].position
                - trace_model.bones[up_from_name].position
            ).normalized()
            bone_cross = (
                trace_model.bones[cross_to_name].position
                - trace_model.bones[cross_from_name].position
            ).normalized()
            bone_cross_vec: MVector3D = bone_up.cross(bone_cross).normalized()

            initial_qq = MQuaternion.from_direction(bone_direction, bone_cross_vec)

            direction_from_abs_pos = poses_mov_motion.bones[direction_from_name][
                mov_bf.index
            ].position
            direction_to_abs_pos = poses_mov_motion.bones[direction_to_name][
                mov_bf.index
            ].position
            direction: MVector3D = (
                direction_to_abs_pos - direction_from_abs_pos
            ).normalized()

            up_from_abs_pos = poses_mov_motion.bones[up_from_name][
                mov_bf.index
            ].position
            up_to_abs_pos = poses_mov_motion.bones[up_to_name][mov_bf.index].position
            up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

            cross_from_abs_pos = poses_mov_motion.bones[cross_from_name][
                mov_bf.index
            ].position
            cross_to_abs_pos = poses_mov_motion.bones[cross_to_name][
                mov_bf.index
            ].position
            cross: MVector3D = (cross_to_abs_pos - cross_from_abs_pos).normalized()

            motion_cross_vec: MVector3D = up.cross(cross).normalized()
            motion_qq = MQuaternion.from_direction(direction, motion_cross_vec)

            cancel_qq = MQuaternion()
            for cancel_name in cancel_names:
                cancel_qq *= poses_rot_motion.bones[cancel_name][mov_bf.index].rotation

            bf = VmdBoneFrame(name=target_bone_name, index=mov_bf.index, register=True)
            bf.rotation = (
                cancel_qq.inverse() * motion_qq * initial_qq.inverse() * invert_qq
            )

            poses_rot_motion.append_bone_frame(bf)

    VmdWriter(
        poses_rot_motion,
        os.path.join(args.out_folder, "output_poses_rot.vmd"),
        "4D-Humans",
    ).save()
