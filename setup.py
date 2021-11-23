from setuptools import setup

setup(
    name='visualize_pyphi',
    version='0.1.0',    
    description='Create beautiful cause effect structure visualizations',
    url='https://github.com/matteograsso/visualize_pyphi',
    author='Matteo Grasso and Bjørn E juel',
    author_email='bjorneju@gmail.com',
    license='BSD 2-clause',
    packages=['visualize_pyphi'],
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
    ],
)
