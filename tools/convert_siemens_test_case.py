import numpy as np
import os
import struct
import sys

from perception import DepthImage
from visualization import Visualizer2D as vis2d

if __name__ == '__main__':
    filename = sys.argv[1]

    filepath, root = os.path.split(filename)
    name, ext = os.path.splitext(root)
    out_filename = os.path.join(filepath, name + '.npy')
    print(out_filename)
    
    depth_map = np.zeros((256*256), dtype=int)

    with open(filename, "rb") as binary_file:
        data = binary_file.read()
        i = 0
        j = 0
        bpp = 2
        while i<(len(data)/bpp):
            depth_map[i] = struct.unpack('<H', data[j:j+bpp])[0]
            
            i = i + 1
            j = j + bpp
            

    depth_map = np.reshape(depth_map, (256, 256)).astype(np.float32) / 1000.0
    depth_im = DepthImage(depth_map)
    depth_im = depth_im.inpaint(rescale_factor=0.25)
    depth_im = depth_im.resize(0.25, interp='nearest')
    
    #vis2d.figure()
    #vis2d.imshow(depth_im)
    #vis2d.show()

    depth_im.save(out_filename)
