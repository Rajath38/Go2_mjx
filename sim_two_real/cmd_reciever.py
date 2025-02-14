from inter_process_com import publisher as pub

# Initialize publisher
PJ = pub.publish_cmd()


while True:
    cmd = PJ.get()['XYyaw']
    print(f"cmd X: {cmd[0]}, Y: {cmd[1]}, Yaw: {cmd[2]}")