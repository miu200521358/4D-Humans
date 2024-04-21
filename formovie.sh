clear

names=("snobbism_cut")

for name in ${names[@]}; do
    echo "=================================="
    echo $name
    # 姿勢推定
    python demo3.py --video /mnt/c/MMD/mmd-auto-trace-4/inputs/$name.mp4 --output_dir demo_outputs --batch_size 48
    # 回転
    echo "------------- 回転-----------------"
    python output4.py --target_dir demo_outputs/$name
    # 足IK
    echo "------------- 足IK-----------------"
    python output5.py --target_dir demo_outputs/$name
    # 腕捩
    echo "------------- 腕捩-----------------"
    python output6.py --target_dir demo_outputs/$name
    # 接地
    echo "------------- 接地-----------------"
    python output8.py --target_dir demo_outputs/$name
    # 間引き
    echo "------------- 間引き -----------------"
    python output9.py --target_dir demo_outputs/$name
done

