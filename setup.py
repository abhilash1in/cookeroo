# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# 'setup.py publish' shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    sys.exit()

packages = ['cookeroo']

requires = [
    'pydub',
    'librosa',
    'numpy',
    'scipy',
    'pandas',
    'keras',
    'tensorflow'
]
test_requirements = [
    'flake8',
    'pytest'
]

about = {}
with open(os.path.join(here, 'cookeroo', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()
with open('LICENSE', 'r', encoding='utf-8') as f:
    license_str = f.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    packages=packages,
    package_data={'': ['LICENSE']},
    package_dir={'cookeroo': 'cookeroo'},
    include_package_data=True,
    python_requires=">=2.7",
    install_requires=requires,
    license=license_str,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
    ],
    # cmdclass={'test': PyTest},
    tests_require=test_requirements,
    project_urls={
        'Documentation': '',
        'Source': 'https://github.com/cookeroo/cookeroo',
    },
)
