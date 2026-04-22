from setuptools import setup, find_packages

install_requires = [
    # TensorFlow
    'tensorflow==2.13.0; python_version >= "3.9" and python_version < "3.12"',
    'tensorflow; python_version >= "3.12"',

    # OpenCV
    'opencv-python>=4.6.0; python_version >= "3.9" and python_version < "3.12"',
    'opencv-python; python_version >= "3.12"',

    # NumPy (tensorflow==2.13.0 상한 준수: <=1.24.3)
    'numpy>=1.23.0,<=1.24.3; python_version >= "3.9" and python_version < "3.12"',
    'numpy; python_version >= "3.12"',

    # gdown
    'gdown',

    # matplotlib
    'matplotlib'
]

setup(
    name='oneclickai',
    version='0.1.13',
    description='OneclickAI package for learning AI with python',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Seung Oh',
    author_email='osy044@naver.com',
    url='https://oneclickai.co.kr',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    keywords=['oneclick', 'clickai', 'learning ai', 'ai model', 'ai', 'ai package', 'oneclickai', 'oneclickai package'],
    python_requires='>=3.9',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)