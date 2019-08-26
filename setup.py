from setuptools import setup
from autoopt.version import AUTOOPT_VERSION


setup(name='autoopt',
    version=AUTOOPT_VERSION,
    description='Automated Optimization for Deep Learning',
    url='https://github.corp.ebay.com/MLOpt/AutoOpt.git',
    author='Selcuk Kopru, Tomer Lancewicki',
    author_email='skopru@ebay.com',
    packages=['autoopt', 'autoopt.optim', 'autoopt.util'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: ML Researchers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache License, Version 2.0',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[],
    setup_requires=[],
    zip_safe=False
)
