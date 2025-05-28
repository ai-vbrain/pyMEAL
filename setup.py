__authors__ = 'Adeleke Maradesa, Abdulmojeed Ilyas'

__date__ = '25th May, 2025'

import setuptools
from os.path import exists, join

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''

dependencies = [
    "tensorflow==2.9.0",
    "matplotlib==3.9.2",
    "SimpleITK==2.4.0",
    "scipy==1.13.1",
    "antspyx==0.5.4",
    "numpy==1.23.5",
    "imageio==2.36.0",
    "nibabel==5.3.2",
    "pillow==11.0.0",
]

setuptools.setup(
    name = "pyMEAL",
    version = "1.0.2",
    author = "The Hong Kong Center for Cerebrocardivascular Health Engineering (COCHE)",
    author_email = "amaradesa@connect.ust.hk",
    description = "pyMEAL: Multi-Encoder-Augmentation-Aware-Learning",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    ###
    url = "https://github.com/ai-vbrain/pyMEAL",
    project_urls = {
        "Source Code": "https://github.com/ai-vbrain/pyMEAL",
        "Bug Tracker": "https://github.com/ai-vbrain/pyMEAL/issues",
    },
    install_requires=dependencies,

    python_requires = ">=3",
    
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
          
    ],
    packages=['pyMEAL'],
    include_package_data=True,
    package_data={'pyMEAL': ['CTScan data/*']}, 
)
