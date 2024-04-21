from glob import glob
import cv2
from tqdm import tqdm


if __name__ == "__main__":
    out = cv2.VideoWriter(
        "C:/MMD/4D-Humans/demo_outputs/snobbism_cut/snobbism_cut3.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (256, 768),
    )

    for img_path in tqdm(sorted(glob("C:/MMD/4D-Humans/demo_outputs/snobbism_cut/image/*.png"))):
        img = cv2.imread(img_path)

        # 画像を左右半分に分ける
        img1 = img[:, :256]
        img2 = img[:, 256:512]
        img3 = img[:, 512:768]
        img4 = img[:, 768:]

        # 縦方向に繋げる
        img2 = cv2.vconcat([img2, img3, img4])

        out.write(img2)

    out.release()
