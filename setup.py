import setuptools


VERSION = "0.0.1"
NAME = "YoloLib"

README_PATH = "./README.md"
SRC_PATH = "./yolo_lib"

with open(README_PATH, "r") as f:
    long_description = f.read()


packages = setuptools.find_packages()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author="Markus Tiller",
    author_email="markustiller11@gmail.com",
    description="Wheel package for vessel detection experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages,
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.8', 
)

