import argparse
import os
import numpy as np
from tqdm import tqdm

from mlib.vmd.vmd_collection import VmdMotion
from mlib.vmd.vmd_part import VmdBoneFrame
from mlib.vmd.vmd_reader import VmdReader
from mlib.vmd.vmd_writer import VmdWriter
from mlib.core.math import MVector3D, MQuaternion
from mlib.core.interpolation import create_interpolation, get_infections2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMR2 demo code")
    parser.add_argument("--target_dir", type=str)

    args = parser.parse_args()

    for person_id in range(10):
        motion_path = os.path.join(
            args.target_dir, f"output_poses_mp_rot_ik_arm_ik_ground_{person_id:02d}.vmd"
        )
        if not os.path.exists(motion_path):
            break

        mp_rot_motion = VmdReader().read_by_filepath(motion_path)
        reduce_motion = VmdMotion()

        xs = []
        ys = []
        zs = []

        for fno in tqdm(range(mp_rot_motion.bones.max_fno)):
            xs.append(mp_rot_motion.bones["センター"][fno].position.x)
            ys.append(mp_rot_motion.bones["センター"][fno].position.y)
            zs.append(mp_rot_motion.bones["センター"][fno].position.z)

        x_infections = get_infections2(xs, 0.05, 0.05)
        y_infections = get_infections2(ys, 0.05, 0.05)
        z_infections = get_infections2(zs, 0.05, 0.05)

        # print("x_infections:")
        # print(x_infections)
        # print("y_infections:")
        # print(y_infections)
        # print("z_infections:")
        # print(z_infections)

        xz_infections = np.array(list(set(x_infections) | set(z_infections)))
        xz_infections.sort()

        y1bf = VmdBoneFrame(name="グルーブ", index=0, register=True)
        y1bf.position = MVector3D(0, ys[0], 0)
        reduce_motion.append_bone_frame(y1bf)

        xz1bf = mp_rot_motion.bones["センター"][0]
        xz1bf.position = MVector3D(xs[0], 0, zs[0])
        reduce_motion.append_bone_frame(xz1bf)

        # グルーブ
        for y1, y2 in tqdm(zip(y_infections[:-1], y_infections[1:])):
            ybz = create_interpolation(ys[y1 : (y2 + 1)])

            y2bf = VmdBoneFrame(name="グルーブ", index=y2, register=True)
            y2bf.position = MVector3D(0, ys[y2], 0)
            y2bf.interpolations.translation_y = ybz

            reduce_motion.append_bone_frame(y2bf)

        # センター
        for x1, x2 in tqdm(zip(xz_infections[:-1], xz_infections[1:])):
            xbz = create_interpolation(xs[x1 : (x2 + 1)])
            zbz = create_interpolation(zs[x1 : (x2 + 1)])

            x2bf = mp_rot_motion.bones["センター"][x2]
            x2bf.position = MVector3D(xs[x2], 0, zs[x2])
            x2bf.interpolations.translation_x = xbz
            x2bf.interpolations.translation_z = zbz

            reduce_motion.append_bone_frame(x2bf)

        left_leg_ik_bf = mp_rot_motion.bones["左足ＩＫ"][0]
        reduce_motion.append_bone_frame(left_leg_ik_bf)

        right_leg_ik_bf = mp_rot_motion.bones["右足ＩＫ"][0]
        reduce_motion.append_bone_frame(right_leg_ik_bf)

        for bone_name in tqdm(["左足ＩＫ", "右足ＩＫ"]):
            xs = []
            ys = []
            zs = []

            rs = [mp_rot_motion.bones[bone_name][0].rotation]
            ds = [1.0]

            for fno in range(mp_rot_motion.bones.max_fno):
                xs.append(mp_rot_motion.bones[bone_name][fno].position.x)
                ys.append(mp_rot_motion.bones[bone_name][fno].position.y)
                zs.append(mp_rot_motion.bones[bone_name][fno].position.z)

                if fno > 0:
                    ds.append(
                        MQuaternion.dot(rs[-1], mp_rot_motion.bones[bone_name][fno].rotation)
                    )
                    rs.append(mp_rot_motion.bones[bone_name][fno].rotation)

            x_infections = get_infections2(xs, 0.1, 0.1)
            y_infections = get_infections2(ys, 0.1, 0.1)
            z_infections = get_infections2(zs, 0.1, 0.1)
            d_infections = get_infections2(ds, 0.0004, 0.0004)

            infections = np.array(
                list(set(x_infections) | set(y_infections) | set(z_infections) | set(d_infections))
            )
            infections.sort()

            for l1, l2 in zip(infections[:-1], infections[1:]):
                xbz = create_interpolation(xs[l1 : (l2 + 1)])
                ybz = create_interpolation(ys[l1 : (l2 + 1)])
                zbz = create_interpolation(zs[l1 : (l2 + 1)])
                rbz = create_interpolation(ds[l1 : (l2 + 1)])

                bf = mp_rot_motion.bones[bone_name][l2]
                bf.interpolations.translation_x = xbz
                bf.interpolations.translation_y = ybz
                bf.interpolations.translation_z = zbz
                bf.interpolations.rotation = rbz

                reduce_motion.append_bone_frame(bf)

        for bone_name in tqdm(mp_rot_motion.bones.names):
            if bone_name in ["センター", "グルーブ", "左足ＩＫ", "右足ＩＫ"]:
                continue

            rs = [mp_rot_motion.bones[bone_name][0].rotation]
            ds = [1.0]
            for fno in range(1, mp_rot_motion.bones.max_fno):
                ds.append(
                    MQuaternion.dot(rs[-1], mp_rot_motion.bones[bone_name][fno].rotation)
                )
                rs.append(mp_rot_motion.bones[bone_name][fno].rotation)

            d_infections = get_infections2(ds, 0.0004, 0.0004)

            r1bf = mp_rot_motion.bones[bone_name][0]
            r1bf.register = True
            reduce_motion.append_bone_frame(r1bf)

            for d1, d2 in zip(d_infections[:-1], d_infections[1:]):
                rbz = create_interpolation(ds[d1 : (d2 + 1)])

                r2bf = mp_rot_motion.bones[bone_name][d2]
                r2bf.interpolations.rotation = rbz

                reduce_motion.append_bone_frame(r2bf)

        VmdWriter(
            reduce_motion,
            os.path.join(
                args.target_dir, f"output_poses_mp_rot_ik_arm_ik_ground_reduce_{person_id:02d}.vmd"
            ),
            "4D-Humans",
        ).save()
