from setuptools import setup

setup(
    name='VoltageGeneration',
    version='1.0',
    scripts=['VoltageGeneration.py'],
    install_requires=[
        'numpy>=1.14',
        'scipy>=1.0.0',
    ],
)

