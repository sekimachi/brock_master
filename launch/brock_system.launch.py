from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        # =====================================================
        # brock_master
        # =====================================================

        Node(
            package="brock_master",
            executable="brock_master",
            name="brock_master",
            output="screen",
        ),

        # =====================================================
        # brock_operate
        # =====================================================

        Node(
            package="brock_operate",
            executable="brock_operate",
            name="brock_operate",
            output="screen",
        ),

    ])
