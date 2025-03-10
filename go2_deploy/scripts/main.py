from go2 import LEGGED_GYM_ROOT_DIR

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from go2_deploy.common.remote_controller import KeyMap
from go2_deploy.config import Config

from go2_deploy.scripts.sense import sense_main
from go2_deploy.scripts.est import est_main
import multiprocessing as mp
import os



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

    sensor_process = mp.Process(target=sense_main, args=config, daemon=True)
    est_process = mp.Process(target=est_main, args=config, daemon=True)


    # Clear the terminal
    os.system("clear")


    #----------------------------------Sensor Read ---------------------------------- 
    sensor_process.start()

    #----------------------------------Run Estimator -------------------------------
    est_process.start()

    #---------------------------------Main COntroller----------------------------



   
    print("Exit")
