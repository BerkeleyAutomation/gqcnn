 test the fcgqcnn on a single object
fcgqcnn_single_object:
        python examples/policy.py FC-GQCNN-4.0-PJ --fully_conv --depth_image data/examples/single_object/primesense/depth_7.npy --segmask data/examples/single_object/primesense/segmask_7.png --camera_intr data/calib/phoxi/phoxi.intr

# test the gqcnn on a single object
gqcnn_single_object_docker:
        python examples/policy.py GQ-Image-Wise --model_dir /root/Workspace/gqcnn/models --depth_image data/examples/single_object/primesense/depth_8.npy --segmask data/examples/single_object/primesense/segmask_8.png --camera_intr data/calib/phoxi/phoxi.intr

# test the gqcnn on a single object
gqcnn_single_object:
        python examples/policy.py GQ-Image-Wise --depth_image data/examples/single_object/primesense/depth_8.npy --segmask data/examples/single_object/primesense/segmask_8.png --camera_intr data/calib/phoxi/phoxi.intr

vir_gqcnn_single_object: 
        /home/neal/Desktop/Codebases/gqcnn/20231227_venv/bin/python examples/policy.py GQ-Image-Wise --depth_image data/examples/single_object/primesense/depth_8.npy --segmask data/examples/single_object/primesense/segmask_8.png --camera_intr data/calib/phoxi/phoxi.intr
