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
    "matplotlib",  
    "tensorflow",  
    "SimpleITK",
    "numpy",
    "scipy",
    "antspyx",
    "PIL",
    "nibabel",
    "imageio",
]

setuptools.setup(
    name = "pyMEAL",
    version = "1.0.1",
    author = "The Hong Kong Center for Cerebrocardivascular Health Engineering (COCHE) and AI-vBRAIN ",
    author_email = "amaradesa@connect.ust.hk",
    description = "pyMEAL: Multi-Encoder-Augmentation-Aware-Learning",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    ###
    url = "https://github.com/ai-vbrain/Multi-Encoder-Augmentation-Aware-Learning",
    project_urls = {
        "Source Code": "https://github.com/ai-vbrain/Multi-Encoder-Augmentation-Aware-Learning",
        "Bug Tracker": "https://github.com/ai-vbrain/Multi-Encoder-Augmentation-Aware-Learning/issues",
    },
    install_requires=dependencies,

    python_requires = ">=3",
    
    classifiers = [
        "Programming Language :: Python :: 2.9.0",
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
