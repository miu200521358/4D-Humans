import argparse
from glob import glob
import json
import os
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame
from mlib.vmd.vmd_writer import VmdWriter
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_collection import PmxModel
from mlib.core.math import MVector3D, MQuaternion

# 身長158cmプラグインより
MIKU_CM = 0.1259496

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

PMX_CONNECTIONS = {
    'Right Ankle': "右足首",
    'Right Knee': "右ひざ",
    'Right Hip': "右足",
    'Left Hip': "左足",
    'Left Knee': "左ひざ",
    'Left Ankle': "左足首",
    'Right Wrist': "右手首",
    'Right Elbow': "右ひじ",
    'Right Shoulder': "右腕",
    'Left Shoulder': "左腕",
    'Left Elbow': "左ひじ",
    'Left Wrist': "左手首",
    'Neck (LSP)': "首",
    'Top of Head (LSP)': "頭先",
    'Pelvis (MPII)': "下半身",
    'Thorax (MPII)': "上半身2",
    'Spine (H36M)': "上半身",
    'Head (H36M)': "頭",
    'Left Eye': "左目",
    'Right Eye': "右目",
    'Left Ear': "左耳",
    'Right Ear': "右耳",
    "Pelvis2": "下半身2",
}

VMD_CONNECTIONS = {
    "下半身": {
        "direction": ("下半身", "下半身2"),
        "up": ("左足", "右足"),
        "cancel": (),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(10, 0, 0),
        },
    },
    "上半身": {
        "direction": ("上半身", "上半身2"),
        "up": ("左腕", "右腕"),
        "cancel": (),
        "invert": {
            "before": MVector3D(10, 0, 0),
            "after": MVector3D(),
        },
    },
    "上半身2": {
        "direction": ("上半身2", "首"),
        "up": ("左腕", "右腕"),
        "cancel": ("上半身",),
        "invert": {
            "before": MVector3D(20, 0, 0),
            "after": MVector3D(),
        },
    },
    "首": {
        "direction": ("上半身2", "首"),
        "up": ("頭", "鼻"),
        "cancel": (
            "上半身",
            "上半身2",
        ),
        "invert": {
            "before": MVector3D(20, 10, 0),
            "after": MVector3D(),
        },
    },
    "頭": {
        "direction": ("頭", "鼻"),
        "up": ("左耳", "右耳"),
        "cancel": (
            "上半身",
            "上半身2",
            "首",
        ),
        "invert": {
            "before": MVector3D(-20, 0, 0),
            "after": MVector3D(),
        },
    },
    "左肩": {
        "direction": ("左肩", "左腕"),
        "up": ("上半身2", "首"),
        "cancel": ("上半身", "上半身2"),
        "invert": {
            "before": MVector3D(0, 0, -20),
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
            "before": MVector3D(0, 0, 20),
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
        "up": ("左足", "右足"),
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
        "up": ("左足", "右足"),
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')

    args = parser.parse_args()

    poses_mov_motion = VmdMotion()
    poses_rot_motion = VmdMotion()

    pmx_reader = PmxReader()
    trace_model = pmx_reader.read_by_filepath(
        "/mnt/c/MMD/mmd-auto-trace-3/data/pmx/trace_model.pmx"
    )

    with open(os.path.join(args.out_folder, "out.json"), 'r') as f:
        out = json.load(f)

    for i, json_path in enumerate(glob(os.path.join(args.out_folder, "*.json"))):
        for jname, joint in joints.items():
            if jname not in PMX_CONNECTIONS:
                pose_bf = VmdBoneFrame(i, jname, register=True)
            else:
                pose_bf = VmdBoneFrame(i, PMX_CONNECTIONS[jname], register=True)
            pose_bf.position = MVector3D(joint['x'], -joint['y'], joint['z'])
            pose_bf.position *= 10

            poses_mov_motion.append_bone_frame(pose_bf)

        pelvis2_bf = VmdBoneFrame(i, "下半身2", register=True)
        pelvis2_bf.position = (
            poses_mov_motion.bones["左ひざ"][i].position
            + poses_mov_motion.bones["右ひざ"][i].position
        ) / 2
        poses_mov_motion.append_bone_frame(pelvis2_bf)

        pelvis_bf = poses_mov_motion.bones["下半身"][i]
        center_bf = VmdBoneFrame(i, "センター", register=True)
        center_bf.position = pelvis_bf.position.copy() / MIKU_CM - trace_model.bones["首"].position
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
