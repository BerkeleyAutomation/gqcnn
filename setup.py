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
	'core',
	'perception',
	'scikit-image',
	'scikit-learn'
]

setup(name='gqcnn', 
	  version='0.1.0', 
	  description='GQCNN project code', 
	  author='Vishal Satish', 
	  author_email='vsatish@berkeley.edu', 
	  package_dir={'': '.'}, 
	  packages=['gqcnn'], 
	  install_requires=requirements)