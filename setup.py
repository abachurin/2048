from setuptools import setup, find_packages

with open('requirements.txt') as fh:
    install_requires = fh.read().split('\n')

setup(name='game_2048',
		version='2.0',
		description='2048 Reinforcement Learning Agent, with dash UI',
		author='Alex Bachurin',
		author_email='bachurin.alex@gmail.com',
		python_requires='>=3.9',
		packages=find_packages(),
		include_package_data=True,
		data_files=[('', ['requirements.txt'])],
    	install_requires=install_requires[:-1],
     )
