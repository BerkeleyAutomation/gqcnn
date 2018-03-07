"""
Setup of gqcnn python codebase
Author: Vishal Satish
"""
from setuptools import setup

requirements = [
	'tensorflow>=1.0',
	'numpy',
	'matplotlib',
	'opencv-python',
	'scipy',
	'scikit-image',
	'scikit-learn'
]

setup(name='gqcnn', 
	  version='0.1.0', 
	  description='GQCNN project code', 
	  author='Vishal Satish', 
	  author_email='vsatish@berkeley.edu', 
	  package_dir={'': '.'}, 
	  packages=['gqcnn', 'gqcnn.model', 'gqcnn.model.tf', 'gqcnn.model.neon', 'gqcnn.training', 'gqcnn.training.tf', 'gqcnn.training.neon', 'gqcnn.utils'], 
	  install_requires=requirements)
