from setuptools import setup, find_packages

setup(
    name='rlbox',
    version='0.0.1',
    description='Reinforcement learning for OpenAI Gym and TensorFlow',
    url='https://github.com/apparatusbox/rlbox',
    author='lgvaz',
    author_email='lucasgouvaz@gmail.com',
    packages=[package for package in find_packages() if package.startswith('rlbox')],
    zip_safe=False)
