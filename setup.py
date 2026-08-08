from setuptools import find_packages, setup

package_name = 'brock_master'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['*.pt'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ( "share/ament_index/resource_index/packages", ["resource/" + package_name], ),
        ( "share/" + package_name, ["package.xml"], ), 
        ( "share/" + package_name + "/launch", ["launch/brock_system.launch.py"], ),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='imamura',
    maintainer_email='sekimachi.287@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
entry_points={
    'console_scripts': [
        'brock_master = brock_master.brock_master:main',
    ],
},
    
)
