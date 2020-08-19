# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = [r.strip() for r in f]

setup(
    name='carla_real_traffic_scenarios',
    version='1.0.0',
    long_description=readme,
    author='Anonymous (for now)',
    author_email='Anonymous (for now)',
    url='https://sites.google.com/view/carla-real-traffic-scenarios/home',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True
)

