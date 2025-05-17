from setuptools import setup, find_packages

install_requires = [
    # TensorFlow
    'tensorflow==2.13.0; python_version < "3.12"',
    'tensorflow; python_version >= "3.12"',
    
    # PyTorch
    'torch==2.1; python_version < "3.12"',
    'torch; python_version >= "3.12"',

    # TorchVision
    'torchvision==0.16; python_version < "3.12"',
    'torchvision; python_version >= "3.12"',

    # TorchAudio
    'torchaudio==2.1; python_version < "3.12"',
    'torchaudio; python_version >= "3.12"',

    # OpenCV
    'opencv-python==4.10.0.84; python_version < "3.12"',
    'opencv-python; python_version >= "3.12"',

    # NumPy
    'numpy==1.24.3; python_version < "3.12"',
    'numpy; python_version >= "3.12"',

    # Pandas
    'pandas==2.2.3; python_version < "3.12"',
    'pandas; python_version >= "3.12"',

    # Ultralytics (YOLO)
    'ultralytics==8.3.28; python_version < "3.12"',
    'ultralytics; python_version >= "3.12"',

    # openpyxl
    'openpyxl',

    # gdown
    'gdown'
]

setup(
    name='oneclickai',
    version='0.1.3',
    description='OneclickAI package for learning AI with python',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Seung Oh',
    author_email='osy044@naver.com',
    url='https://oneclickai.co.kr',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    keywords=['oneclick', 'clickai', 'learning ai', 'ai model', 'ai', 'ai package', 'oneclickai', 'oneclickai package'],
    python_requires='>=3.7',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)