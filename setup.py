from setuptools import setup, find_packages

setup(name='gymmeforce',
      version='0.0.1',
      description='Reinforcement learning for OpenAI Gym and TensorFlow',
      url='https://github.com/lgvaz/gymmeforce',
      author='lgvaz',
      author_email='lucasgouvaz@gmail.com',
      packages=[package for package in find_packages() if package.startswith('gymmeforce')],
      zip_safe=False)
