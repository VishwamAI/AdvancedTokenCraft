from setuptools import setup, find_packages

setup(
    name='AdvancedTokenCraft',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
    ],
    entry_points={
        'console_scripts': [
            'advancedtokencraft=AdvancedTokenCraft.__main__:main',
        ],
    },
)
