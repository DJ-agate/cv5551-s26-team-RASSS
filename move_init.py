from xarm.wrapper import XArmAPI




#for robot frame
TAG_SIZE = 0.08
GRIPPER_LENGTH = 0.1 *1000 #0.069 * 1000
TCP_OFFSET = [0,-5,GRIPPER_LENGTH,0,0,0]


CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0205

robot_ip = '192.168.1.172'
INIT_POSE = [86.954865, 0.818719, 85.287003, 179.991483, -0.001547, 0.001776]


def main():

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)

    arm.set_mode(0)
    arm.set_state(0)
    arm.set_position(245,100,45,180,0,0,is_radian=None,wait=True) #obstacle tests
    # arm.set_position(245,0,45,180,0,0,is_radian=None,wait=True) #side tests
    # arm.set_position(100,0,45,180,0,0,is_radian=None,wait=True) # center workspace tests




    arm.disconnect()


if __name__ == "__main__":
    main()
