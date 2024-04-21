C:\MMD\4D-Humans>

conda env create -f environment.yml
conda activate 4dhumans

python demo.py --img_folder example_data/images --out_folder demo_out --batch_size=48 --side_view --save_mesh --full_frame

pip install PyOpenGL PyOpenGL_accelerate






sudo apt-get install libx11-dev libxrandr-dev libxcursor-dev libxxf86vm-dev libxinerama-dev libxi-dev libglu1-mesa-dev libglew-dev libgles2-mesa-dev



wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1


conda create --name 4dhumans python=3.10
conda activate 4dhumans
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install numpy pytorch-lightning smplx==0.1.28 pyrender opencv-python yacs scikit-image einops timm dill pandas
pip install git+https://github.com/facebookresearch/detectron2
pip install git+https://github.com/mattloper/chumpy
pip install -e .

python demo.py --img_folder example_data/images --out_folder demo_out --batch_size=48 --side_view --save_mesh --full_frame


conda remove -n 4dhuman --all


cd /mnt/c/MMD/4D-Humans/
conda env create -f environment.yml
conda activate 4dhumans
pip install webdataset
pip install git+https://github.com/brjathu/PHALP.git



class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        cfg.render.enable = False
        super().__init__(cfg)

python demo.py --img_folder example_data/snobbism_1080-1380/images2 --out_folder example_data/snobbism_1080-1380/outputs2 --batch_size 48 --side_view

python output.py --out_folder example_data/snobbism_1080-1380/outputs2

python track.py video.source="example_data/snobbism_1080-1380/images2"

export PATH=/home/miu/anaconda3/envs/4dhumans/bin:$PATH

