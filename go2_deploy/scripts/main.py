from go2 import LEGGED_GYM_ROOT_DIR

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from go2_deploy.common.remote_controller import KeyMap
from go2_deploy.config import Config

from go2_deploy.scripts.sense import Sense



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{args.config}"
   
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)


    #----------------------------------Sensor read ---------------------------------- 
    
    sensor_read = Sense(config)
    
    #start collecting sensor data
    sensor_read.start()


    #----------------------------------Run Estimator -------------------------------



    while True: 
        try:
            # Press the select key to exit
            if sensor_read.remote_controller.button[KeyMap.X] == 1:
                break
        except KeyboardInterrupt:
            break     
   
    print("Exit")
