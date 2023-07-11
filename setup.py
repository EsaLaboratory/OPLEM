from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
  
setup(
    name='oplem',
    version='0.0.1',
    description='Open Platform for Local Energy Markets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EsaLaboratory/oplem',
    author='Chaimaa Essayeh',
    author_email='cessayeh@ed.ac.uk',
    license='MIT',
    packages=['oplem'], #same as name
    include_package_data=True,
    package_dir={"": "oplem/Data"},
    install_requires=[
        'cvxopt',
        'notebook',
        'matplotlib',
        'numpy',
        'pandapower',
        'pandas',
        'PICOS',
        'scikit-learn',
        'scipy'

    ],
)
