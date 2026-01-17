from setuptools import setup

setup(name='pchem',
      version='0.3.1',
      description='Physical Chemistry helper functions, scripts and tools.',
      url='https://github.com/ryanpdwyer/pchem',
      author='Ryan Dwyer',
      author_email='dwyerry@mountunion.edu',
      license='MIT',
      packages=['pchem', 'pchemapps', 'pages', 'pages.kinetics', 'pages.thermodynamics', 'pages.quantum', 'pages.data_analysis', 'pages.electrochemistry', 'pages.ai_tools', 'pages.utilities'],
      include_package_data=True,
      install_requires=[
          'sympy',
      ],
      zip_safe=False)
