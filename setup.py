from setuptools import setup, find_packages

setup(
    name="walkg",
    version="0.1.0",
    author="Vitor Marques",
    license='MIT',
    packages=find_packages(include=["walkg", "walkg.*"]),
    install_requires=["scipy", "numpy"]
)
