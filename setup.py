# mdp/setup.py

from setuptools import setup, find_packages

setup(name='mdp', version='1.0', packages=find_packages(),
      install_requires=['statsmodels', 'pandas', 'matplotlib', 'numpy', 'seaborn', 'scikit-learn', 'tensorflow'])

# __EOF__
