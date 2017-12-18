"""
Setup of gqcnn python codebase
Author: Vishal Satish
"""
from setuptools import setup

requirements = [
    'autolab_core',
    'autolab_perception',
    'numpy',
    'scipy',
    'matplotlib',
    'opencv-python',
    'tensorflow>=1.0',
    'ipython',
    'scikit-image',
    'scikit-learn'
]

exec(open('gqcnn/version.py').read())

setup(name='gqcnn', 
      version=__version__, 
      description='Project code for running Grasp Quality Convolutional Neural Networks', 
      author='Vishal Satish', 
      author_email='vsatish@berkeley.edu', 
      license = 'Apache Software License',
      url = 'https://github.com/BerkeleyAutomation/gqcnn',
      keywords = 'robotics grasping vision deep learning',
      classifiers = [
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering'
      ],      
      packages=['gqcnn'], 
      setup_requres = requirements,
      install_requires = requirements,
      extras_require = { 'docs' : [
          'sphinx',
          'sphinxcontrib-napoleon',
          'sphinx_rtd_theme'
      ],
      }
)
