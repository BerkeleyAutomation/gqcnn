git archive --format=tar -o docker/gqcnn.tar --prefix=gqcnn/ docker
docker build --no-cache -t gqcnn/gpu -f docker/gpu/Dockerfile .
rm docker/gqcnn.tar
