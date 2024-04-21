clear

# names=("snobbism_300-2300" "burai_996-1300" "heart_3039-4838")
# names=("snobbism" "sakura" "late")
# names=("heart_3039-4838" "late" "sakura" "snobbism" "snobbism_300-2300")
# names=("buster" "night")
names=("snobbism_cut")

for name in ${names[@]}; do
    echo "=================================="
    echo $name
    # # 回転
    # echo "------------- 回転-----------------"
    # python output4.py --target_dir demo_outputs/$name
    # 足IK
    # echo "------------- 足IK-----------------"
    # python output5.py --target_dir demo_outputs/$name
    # # 腕捩
    # echo "------------- 腕捩-----------------"
    # python output6.py --target_dir demo_outputs/$name
    # 接地
    echo "------------- 接地-----------------"
    python output8-2.py --target_dir demo_outputs/$name
    # # 間引き
    # echo "------------- 間引き -----------------"
    # python output9.py --target_dir demo_outputs/$name
done

