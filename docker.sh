cd ~/cudaLearn/build
cmake ..
make
docker cp ../bin/test.exe tensorrt:/workspace/test.exe
docker exec -it tensorrt /workspace/test.exe ${1} --matrix-size 1024 2048 16