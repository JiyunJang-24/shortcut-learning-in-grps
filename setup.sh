
# You may use a different PyTorch version, but we recommend this exact combo as others may fail.
# Official guides: https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121


# Enter the project root
cd shortcut-learning-in-grps
pip install -e .

# Install FlashAttention 2 (speeds up attention during training): https://github.com/Dao-AILab/flash-attention
# Tip: if the build fails, try: `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja is available (exit code should be 0)
pip install "flash-attn==2.5.5" --no-build-isolation


# Install the LIBERO project package for dataset generation
cd LIBERO
pip install -e .
cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
pip install opencv-python numpy==1.26.4

# LeRobot (for dataset format conversion)
git clone https://github.com/Lucky-Light-Sun/lerobot.git
cd lerobot 
pip install -e .
pip install draccus==0.8.0 numpy==1.26.4 rerun-sdk==0.23.0
# RLDS dataset builder
cd ..
cd dataset_git
git clone https://github.com/Lucky-Light-Sun/rlds_dataset_builder.git

# VQ-BET (for action tokenization/chunking)
git clone https://github.com/jayLEE0301/vq_bet_official.git
cd vq_bet_official
pip install -r requirements.txt
pip install -e .


# Pin specific package versions for compatibility with our experiments
cd ..
pip install robosuite==1.4.0 transforms3d==0.4.2
pip install torchcodec==0.2.1 diffusers==0.32.2
pip install datasets==3.4.1 huggingface-hub==0.29.2 draccus==0.8.0
pip install imageio==2.37.0
pip install numpy==1.26.4
