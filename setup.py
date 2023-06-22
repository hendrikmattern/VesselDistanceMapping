from setuptools import setup, find_packages
from vdm.version import __version__

REQUIRED_PACKAGES = ['scikit-image>=XXX', 'numpy>=XXX']  # todo

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="VDM",
    version=__version__,
    author="Hendrik Mattern",
    author_email="hendrik.mattern@googlemail.com",  # todo
    description="Package for vessel distance mapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",  # todo
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # todo eg BSD License
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',  # todo
)

