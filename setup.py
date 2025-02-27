from setuptools import setup, find_packages


install_requires=['tensorflow==2.11', 'ultralytics==8.3.28', 'opencv-python==4.10.0.84',],

setup(
    name='oneclickai',
    version='0.0.2',
    description='OneclickAI package for learnig AI with python',
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
        'Programming Language :: Python :: 3.10'
    ],
)
