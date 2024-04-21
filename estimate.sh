clear

names=("buster" "night" "heart")

for name in ${names[@]}; do
    echo "=================================="
    echo $name
    # 姿勢推定
    python demo2.py --video /mnt/c/MMD/mmd-auto-trace-4/inputs/$name.mp4 --output_dir demo_outputs --batch_size 48
done

