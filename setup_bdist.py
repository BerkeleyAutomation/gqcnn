# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Setup of gqcnn python codebase
Author: Vishal Satish
"""
import glob
import os
from setuptools import setup
from setuptools.extension import Extension
import shutil

from Cython.Build import cythonize, build_ext

requirements = [
    'autolab-core',
    'autolab-perception',
    'visualization',
    'numpy',
    'scipy',
    'matplotlib<=2.2.0',
    'opencv-python',
    'tensorflow-gpu',
    'scikit-image',
    'scikit-learn',
    'psutil',
    'gputil'
]

exec(open('gqcnn/version.py').read())

def get_extensions_and_inits(rootdir):

    exts = []
    inits = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        ext_name = dirpath.replace('/', '.')
        has_py = False
        for fn in filenames:
            full_fn = os.path.join(dirpath, fn)
            if fn == '__init__.py':
                inits.append(full_fn)
            if fn[-3:] == '.py':
                has_py = True
        if has_py:
            exts.append(Extension(
                ext_name + '.*',
                [os.path.join(dirpath, '*.py')]
            ))
    return exts, inits

exts, inits = get_extensions_and_inits('gqcnn')

class BuildExtCopyInit(build_ext):

    def run(self):
        super(BuildExtCopyInit, self).run()
        for initfile in inits:
            shutil.copyfile(initfile,
                            os.path.join(str(self.build_lib), initfile))
            initfilec = initfile[:-2] + 'cpython*.so'
            rmfiles = glob.glob(os.path.join(str(self.build_lib), initfilec))
            for rmf in rmfiles:
                os.remove(rmf)

setup(name='gqcnn', 
    version=__version__, 
    description='Project code for running Grasp Quality Convolutional Neural Networks', 
    author='Vishal Satish', 
    author_email='vsatish@berkeley.edu', 
    license = 'Berkeley Copyright',
    url = 'https://github.com/BerkeleyAutomation/gqcnn',
    keywords = 'robotics grasping vision deep learning',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages=[],
    setup_requres = requirements,
    install_requires = requirements,
    extras_require = { 'docs' : [
        'sphinx',
        'sphinxcontrib-napoleon',
        'sphinx_rtd_theme'
    ],
    },
    ext_modules=cythonize(
        exts,
        build_dir='build',
        compiler_directives=dict(
            always_allow_keywords=True
        )
    ),
    cmdclass = {
        'build_ext': BuildExtCopyInit,
    },
)
