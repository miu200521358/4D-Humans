import argparse
from glob import glob
import json
import os
from pykalman import UnscentedKalmanFilter

import numpy as np
from tqdm import tqdm
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame
from mlib.vmd.vmd_writer import VmdWriter
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_collection import PmxModel
from mlib.core.math import MVector3D, MQuaternion

# # 身長158cmプラグインより
# MIKU_CM = 0.1259496

MIKU_CM = 0.09

JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    "OP Nose",  # 0
    "OP Neck",  # 1
    "OP RShoulder",  # 2
    "OP RElbow",  # 3
    "OP RWrist",  # 4
    "OP LShoulder",  # 5
    "OP LElbow",  # 6
    "OP LWrist",  # 7
    "OP MidHip",  # 8
    "OP RHip",  # 9
    "OP RKnee",  # 10
    "OP RAnkle",  # 11
    "OP LHip",  # 12
    "OP LKnee",  # 13
    "OP LAnkle",  # 14
    "OP REye",  # 15
    "OP LEye",  # 16
    "OP REar",  # 17
    "OP LEar",  # 18
    "OP LBigToe",  # 19
    "OP LSmallToe",  # 20
    "OP LHeel",  # 21
    "OP RBigToe",  # 22
    "OP RSmallToe",  # 23
    "OP RHeel",  # 24
    # 24 Ground Truth joints (superset of joints from different datasets)
    "Right Ankle",  # 25
    "Right Knee",  # 26
    "Right Hip",  # 27
    "Left Hip",  # 28
    "Left Knee",  # 29
    "Left Ankle",  # 30
    "Right Wrist",  # 31
    "Right Elbow",  # 32
    "Right Shoulder",  # 33
    "Left Shoulder",  # 34
    "Left Elbow",  # 35
    "Left Wrist",  # 36
    "Neck (LSP)",  # 37
    "Top of Head (LSP)",  # 38
    "Pelvis (MPII)",  # 39
    "Thorax (MPII)",  # 40
    "Spine (H36M)",  # 41
    "Jaw (H36M)",  # 42
    "Head (H36M)",  # 43
    "Nose",  # 44
    "Left Eye",  # 45
    "Right Eye",  # 46
    "Left Ear",  # 47
    "Right Ear",  # 48
]

PMX_CONNECTIONS = {
    "OP Nose": "鼻",  # 0
    "OP Neck": "首",  # 1
    "OP RShoulder": "右腕",  # 2
    "OP RElbow": "右ひじ",  # 3
    "OP RWrist": "右手首",  # 4
    "OP LShoulder": "左腕",  # 5
    "OP LElbow": "左ひじ",  # 6
    "OP LWrist": "左手首",  # 7
    "OP MidHip": "下半身",  # 8
    "OP RHip": "右足",  # 9
    "OP RKnee": "右ひざ",  # 10
    "OP RAnkle": "右足首",  # 11
    "OP LHip": "左足",  # 12
    "OP LKnee": "左ひざ",  # 13
    "OP LAnkle": "左足首",  # 14
    "OP REye": "右目",  # 15
    "OP LEye": "左目",  # 16
    "OP REar": "右耳",  # 17
    "OP LEar": "左耳",  # 18
    "OP LBigToe": "左つま先親",  # 19
    "OP LSmallToe": "左つま先子",  # 20
    "OP LHeel": "左かかと",  # 21
    "OP RBigToe": "右つま先親",  # 22
    "OP RSmallToe": "右つま先子",  # 23
    "OP RHeel": "右かかと",  # 24
    "Pelvis (MPII)": "上半身",  # 39
    "Spine (H36M)": "上半身2",  # 41
    "Head (H36M)": "頭",  # 43
    "Pelvis2": "下半身2",
}

VMD_CONNECTIONS = {
    "下半身": {
        "direction": ("下半身", "下半身2"),
        "up": ("左足", "右足"),
        "cancel": (),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "上半身": {
        "direction": ("上半身", "上半身2"),
        "up": ("左腕", "右腕"),
        "cancel": (),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "上半身2": {
        "direction": ("上半身2", "首"),
        "up": ("左腕", "右腕"),
        "cancel": ("上半身",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "首": {
        "direction": ("首", "頭"),
        "up": ("左腕", "右腕"),
        "cancel": (
            "上半身",
            "上半身2",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "頭": {
        "direction": ("首", "頭"),
        "up": ("左目", "右目"),
        "cancel": (
            "上半身",
            "上半身2",
            "首",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左肩": {
        "direction": ("左肩", "左腕"),
        "up": ("上半身2", "首"),
        "cancel": ("上半身", "上半身2"),
        "invert": {
            "before": MVector3D(0, 0, 20),
            "after": MVector3D(),
        },
    },
    "左腕": {
        "direction": ("左腕", "左ひじ"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左ひじ": {
        "direction": ("左ひじ", "左手首"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
            "左腕",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左手首": {
        "direction": ("左手首", "左中指１"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
            "左腕",
            "左ひじ",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右肩": {
        "direction": ("右肩", "右腕"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
        ),
        "invert": {
            "before": MVector3D(0, 0, -20),
            "after": MVector3D(),
        },
    },
    "右腕": {
        "direction": ("右腕", "右ひじ"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右ひじ": {
        "direction": ("右ひじ", "右手首"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
            "右腕",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右手首": {
        "direction": ("右手首", "右中指１"),
        "up": ("上半身2", "首"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
            "右腕",
            "右ひじ",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左足": {
        "direction": ("左足", "左ひざ"),
        "up": ("左足", "右足"),
        "cancel": ("下半身",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左ひざ": {
        "direction": ("左ひざ", "左足首"),
        "up": ("左足", "右足"),
        "cancel": (
            "下半身",
            "左足",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "左足首": {
        "direction": ("左足首", "左つま先"),
        "up": ("左つま先親", "左つま先子"),
        "cancel": (
            "下半身",
            "左足",
            "左ひざ",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右足": {
        "direction": ("右足", "右ひざ"),
        "up": ("左足", "右足"),
        "cancel": ("下半身",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右ひざ": {
        "direction": ("右ひざ", "右足首"),
        "up": ("左足", "右足"),
        "cancel": (
            "下半身",
            "右足",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "右足首": {
        "direction": ("右足首", "右つま先"),
        "up": ("右つま先親", "右つま先子"),
        "cancel": (
            "下半身",
            "右足",
            "右ひざ",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMR2 demo code")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="demo_out",
        help="Output folder to save rendered results",
    )

    args = parser.parse_args()

    poses_joint_motions: dict[str, VmdMotion] = {}
    poses_number_motions: dict[str, VmdMotion] = {}
    poses_mov_motions: dict[str, VmdMotion] = {}
    poses_rot_motions: dict[str, VmdMotion] = {}

    pmx_reader = PmxReader()
    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-4/configs/pmx/trace_model.pmx"
        # "/mnt/d/MMD/MikuMikuDance_v926x64/UserFile/Model/_あにまさ式ミク準標準見せパン/初音ミクVer2 準標準 見せパン 3_trace.pmx"
    )

    with open(os.path.join(args.target_dir, "out.json"), "r") as f:
        out = json.load(f)

    start_root_poses = {}

    for image_name, frame_out in tqdm(out.items()):
        i = int(image_name)
        root_pos = None
        for person_id, frame_person_out in frame_out.items():
            if person_id not in poses_number_motions:
                poses_number_motions[person_id] = VmdMotion()
                poses_mov_motions[person_id] = VmdMotion()
                poses_rot_motions[person_id] = VmdMotion()
                poses_joint_motions[person_id] = VmdMotion()

            root_pos = MVector3D(
                frame_person_out["cam_t"]["x"],
                -frame_person_out["cam_t"]["y"],
                frame_person_out["cam_t"]["z"],
            ) / MVector3D(MIKU_CM, MIKU_CM, 0.3 / MIKU_CM)
            if person_id not in start_root_poses:
                start_root_poses[person_id] = root_pos.copy()
                start_root_poses[person_id].x = 0.0

            for j, (jname, joint) in enumerate(frame_person_out["joints"].items()):
                if jname not in PMX_CONNECTIONS:
                    pose_bf = VmdBoneFrame(i, jname, register=True)
                else:
                    pose_bf = VmdBoneFrame(i, PMX_CONNECTIONS[jname], register=True)
                pose_bf.position = (
                    MVector3D(joint["x"], -joint["y"], joint["z"]) / MIKU_CM
                )
                pose_bf.position -= start_root_poses[person_id]
                poses_mov_motions[person_id].append_bone_frame(pose_bf)

                number_bf = VmdBoneFrame(i, f"{j}", register=True)
                number_bf.position = pose_bf.position.copy()
                poses_number_motions[person_id].append_bone_frame(number_bf)

                joint_bf = VmdBoneFrame(i, jname, register=True)
                joint_bf.position = pose_bf.position.copy()
                poses_joint_motions[person_id].append_bone_frame(joint_bf)

            pelvis2_bf = VmdBoneFrame(i, "下半身2", register=True)
            pelvis2_bf.position = (
                poses_mov_motions[person_id].bones["左足"][i].position
                + poses_mov_motions[person_id].bones["右足"][i].position
            ) / 2
            poses_mov_motions[person_id].append_bone_frame(pelvis2_bf)

            left_shoulder_bf = VmdBoneFrame(i, "左肩", register=True)
            left_shoulder_bf.position = (
                poses_mov_motions[person_id].bones["首"][i].position
                + poses_mov_motions[person_id].bones["左腕"][i].position
            ) / 2
            poses_mov_motions[person_id].append_bone_frame(left_shoulder_bf)

            right_shoulder_bf = VmdBoneFrame(i, "右肩", register=True)
            right_shoulder_bf.position = (
                poses_mov_motions[person_id].bones["首"][i].position
                + poses_mov_motions[person_id].bones["右腕"][i].position
            ) / 2
            poses_mov_motions[person_id].append_bone_frame(right_shoulder_bf)

            left_toe_bf = VmdBoneFrame(i, "左つま先", register=True)
            left_toe_bf.position = (
                poses_mov_motions[person_id].bones["左つま先親"][i].position
                + poses_mov_motions[person_id].bones["左つま先子"][i].position
            ) / 2
            poses_mov_motions[person_id].append_bone_frame(left_toe_bf)

            right_toe_bf = VmdBoneFrame(i, "右つま先", register=True)
            right_toe_bf.position = (
                poses_mov_motions[person_id].bones["右つま先親"][i].position
                + poses_mov_motions[person_id].bones["右つま先子"][i].position
            ) / 2
            poses_mov_motions[person_id].append_bone_frame(right_toe_bf)

            center_bf = VmdBoneFrame(i, "センター", register=True)
            center_bf.position = root_pos - start_root_poses[person_id]
            poses_rot_motions[person_id].append_bone_frame(center_bf)

            root_bf = VmdBoneFrame(i, "全ての親", register=True)
            root_bf.position = root_pos - start_root_poses[person_id]
            poses_joint_motions[person_id].append_bone_frame(root_bf)

    for (person_id, poses_number_motion), (_, poses_mov_motion), (
        _,
        poses_joint_motion,
    ) in zip(
        poses_number_motions.items(),
        poses_mov_motions.items(),
        poses_joint_motions.items(),
    ):
        VmdWriter(
            poses_mov_motion,
            os.path.join(args.target_dir, f"output_poses_mov_{int(person_id):02d}.vmd"),
            "4D-Humans",
        ).save()

        VmdWriter(
            poses_number_motion,
            os.path.join(
                args.target_dir, f"output_poses_mov_number_{int(person_id):02d}.vmd"
            ),
            "4D-Humans",
        ).save()

        VmdWriter(
            poses_joint_motion,
            os.path.join(
                args.target_dir, f"output_poses_mov_joint_{int(person_id):02d}.vmd"
            ),
            "4D-Humans",
        ).save()

        for target_bone_name, vmd_params in tqdm(VMD_CONNECTIONS.items(), desc="回転"):
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
                up_to_abs_pos = poses_mov_motion.bones[up_to_name][
                    mov_bf.index
                ].position
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
                    cancel_qq *= (
                        poses_rot_motions[person_id]
                        .bones[cancel_name][mov_bf.index]
                        .rotation
                    )

                bf = VmdBoneFrame(
                    name=target_bone_name, index=mov_bf.index, register=True
                )
                bf.rotation = (
                    cancel_qq.inverse() * motion_qq * initial_qq.inverse() * invert_qq
                )

                poses_rot_motions[person_id].append_bone_frame(bf)

        VmdWriter(
            poses_rot_motions[person_id],
            os.path.join(
                args.target_dir, f"output_poses_mp_rot_{int(person_id):02d}.vmd"
            ),
            "4D-Humans",
        ).save()
