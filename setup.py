from setuptools import setup


setup(
    name='psydiff',
    version='0.1',
    author='Yin Wang',
    description=('A structural differencer for Python. '
                 'Parses Python into ASTs, compares them, '
                 'and generates interactive HTML.'),
    packages=['psydiff'],
    package_dir={'psydiff': '.'},
    package_data={'psydiff': ['diff.css', 'nav.js']},
    entry_points={'console_scripts': ['psydiff = psydiff.psydiff:main']},
    license='GNU GPLv3',
    url='https://github.com/yinwang0/psydiff',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved'
        ' :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities'
    ]
)
