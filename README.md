# CVSP
Arial Person Detection and Tracking in Drone Footage

# Training

In order to train the model a pip3 package needs to be installed, based on the available cuda version
1. Run `nvidia-smi`
2. Get CUDA Version e.g. 13.0
3. Run `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`, where CUDA Ver -> cu130