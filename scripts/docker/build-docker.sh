# build the CPU and GPU docker images

git archive --format=tar -o docker/gqcnn.tar --prefix=gqcnn/ master
docker build --no-cache -t gqcnn/gpu -f docker/gpu/Dockerfile .
docker build --no-cache -t gqcnn/cpu -f docker/cpu/Dockerfile .
rm docker/gqcnn.tar
