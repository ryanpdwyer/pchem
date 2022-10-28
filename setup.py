from setuptools import setup

setup(name='pchem',
      version='0.2',
      description='Physical Chemistry helper functions, scripts and tools.',
      url='https://github.com/ryanpdwyer/pchem',
      author='Ryan Dwyer',
      author_email='dwyerry@mountunion.edu',
      license='MIT',
      packages=['pchem'],
      install_requires=[
          'sympy',
      ],
      zip_safe=False)
