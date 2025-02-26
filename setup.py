from setuptools import setup, find_packages

setup(
    name="fluid_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
        "numba>=0.54.0",
    ],
)
