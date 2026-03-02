from pathlib import Path

from setuptools import find_packages

from conda_setup import setup

dirpath = Path(__file__).parent

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == '__main__':
    setup(
        name="openmcadapter",
        version="0.0.1",
        description="Code for runing openmc via the RAMP",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.10',
        requirements_yml='requirements.yml',
    )
