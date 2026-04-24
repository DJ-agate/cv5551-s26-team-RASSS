from xarm.wrapper import XArmAPI

robot_ip = '192.168.1.172'
def main():

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)

    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)




    arm.disconnect()


if __name__ == "__main__":
    main()
