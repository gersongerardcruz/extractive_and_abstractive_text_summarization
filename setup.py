from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A combination of extractive and abstractive text summarization for summarizing long scientific texts',
    author='Gerson Cruz',
    license='MIT',
    python_requires=">=2.7, <3.10"
)
