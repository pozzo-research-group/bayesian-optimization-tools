from setuptools import setup,find_packages

def parse_requirements_file(filename):
    """Read the lines of the requirements file."""
    with open(filename) as input_file:
        return input_file.read().splitlines()

if __name__ == '__main__':
    requirements = parse_requirements_file('requirements.txt')
    install_requires = []
    optional_dependencies = {}
    
    for requirement in requirements:
            install_requires.append(requirement)  
    setup(name="bot",
          description="Bayesian Optimization Tools for Experimental Campaigns",
          version='1.0',
          author='Kiran Vaddi',
          author_email='kiranvad@uw.edu',
          license='MIT',
          python_requires='>=3.11',
          install_requires=install_requires,
          extras_require = {},
          packages=find_packages(),
          long_description=open('README.md').read(),
          long_description_content_type="text/markdown",
          classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
          ],
    )