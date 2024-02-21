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
    package_dir={"": "src"},
    #package_dir={"oplem": "src/oplem"},
    #package_data={"oplem": ["Data/*"]},
    include_package_data=True,
    install_requires=[
        'cvxopt',
        'matplotlib',
        'mosek',
        'notebook',
        'numpy',
        'openpyxl',
        'pandapower',
        'pandas',
        'PICOS',
        'scikit-learn',
        'scipy'
    ],
)
