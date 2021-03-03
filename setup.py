from setuptools import find_packages
from setuptools import setup


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ehds",
    version="0.1",
    author="Brunno Oliveira",
    author_email="brunnokick@gmail.com",
    description="A example package for machine-learning pipelines",
    url="https://github.com/brunnokick/EHDS",
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
)