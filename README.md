

## How to start init the docker
```shell
git clone org-3189299@github.com:Xilinx/Vitis-AI.git

./docker_build.sh -t gpu -f pytorch # for building if gpu nvidia exist

docker images | grep vitis-ai # verify images build exist

cd ..

docker run --gpus all \
  -v ~/.ssh:/home/vitis-ai-user/.ssh:ro \
  -v /home/hadynata/vs-code-data/C++/apollo-project:/home/vitis-ai-user/workspace/apollo-project \
  -v /tools/Xilinx:/tools/Xilinx \
  -v /home/hadynata/kria-vitis-platforms:/home/vitis-ai-user/workspace/kria-vitis-platforms \
  --name apollo_container \
  -it xilinx/vitis-ai-pytorch-gpu:3.5.0.001-77cb9e6ad /bin/bash
 # launch the docker image, only run once, later you just need to stop / start it based on your needs
 
docker start apollo_container // later if wanted to start the container

conda activate vitis-ai-pytorch // in the docker container
```

## How to activate the host and run the accelerator (run this on the KR260)
```shell
git clone git@github.com:BrianHadianSTEI23/apollo-project

cd apollo-project/

dtc -@ -I dts -O dtb -o snn_app.dtbo snn_app.dts

sudo cp snn_app.bit.bin /lib/firmware/xilinx/snn_app/
sudo cp snn_app.dtbo /lib/firmware/xilinx/snn_app/

g++ -O3 -std=c++17 main.cpp -o snn_host -I/usr/include/xrt -lxrt_coreutil -pthread

sudo modprobe zocl
sudo xmutil unloadapp
sudo xmutil loadapp snn_app

sudo ./snn_host snn_kernel.xclbin # run the main program
```


## Furhter resources
https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html
https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Installing-the-Tools // use only the links referred here as the webpage is old
