from setuptools import setup, find_packages
import os

if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ''

requirements = [
    'onnx-tf',
    'onnxruntime',
    'intel-tensorflow',
    'intel-openmp',
    'numpy',
    'scikit-learn',
    'scikit-image',
    'ipython',
    'keras',
    'opencv-python',
    'matplotlib',
    'livelossplot',
    'sympy',
    'scikit-learn',
    'seaborn',
    'tensorflow',
    'PySide6'
]

setup(
    name="Hand Written Math Symbol Evaluation System",
    version="0.1.0",
    author="Domonkos Gyömörey",
    description='Hand Written Math Symbol Evaluation System, using different CNN network',
    long_description=long_description,
    url='https://github.com/domonkosgyomorey/image_to_expression',
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    python_requires=">=3.7"
)