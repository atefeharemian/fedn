from setuptools import find_packages, setup

setup(
    name='fedn',
    version='0.4.1',
    description="""Scaleout Federated Learning""",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Scaleout Systems AB',
    author_email='contact@scaleoutsystems.com',
    url='https://www.scaleoutsystems.com',
    py_modules=['fedn'],
    python_requires='>=3.7,<3.11',
    install_requires=[
        "PyYAML>=5.4",
        "requests",
        "urllib3>=1.26.4",
        "minio",
        "python-slugify",
        "grpcio~=1.47.0",
        "grpcio-tools",
        "numpy>=1.21.6,<=1.22.2",
        "protobuf",
        "pymongo",
        "Flask",
        "Flask-WTF",
        "pyjwt",
        "pyopenssl",
        "ttictoc",
        "psutil",
        "click==8.0.1",
        "jinja2",
        "plotly",
        "pandas",
        "bokeh<3.0.0",
        "networkx"
    ],
    license='Apache 2.0',
    zip_safe=False,
    entry_points={
        'console_scripts': ["fedn=cli:main"]
    },
    keywords='Federated learning',
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
