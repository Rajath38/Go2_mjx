from go2 import LEGGED_GYM_ROOT_DIR

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize

from go2_deploy.common.remote_controller import KeyMap
from go2_deploy.config import Config

from go2_deploy.scripts.sense import sense_main
from go2_deploy.scripts.est import est_main
from go2_deploy.scripts.controller_dummy import controller_main
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
import multiprocessing as mp
import os
import time
import argparse


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="go2.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/go2_deploy/configs/{args.config}"
   
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)
    
    #lowcmd_publisher = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
    mp.set_start_method("spawn", force=True)
 
    sensor_process = mp.Process(target=sense_main, args=(config,), daemon=True)
    #est_process = mp.Process(target=est_main, daemon=True)
    #controller_process = mp.Process(target=controller_main, args=(config, lowcmd_publisher), daemon= True)


    # Clear the terminal
    os.system("clear")


    #----------------------------------Sensor Read ---------------------------------- 
    sensor_process.start()
    time.sleep(5)

    #----------------------------------Run Estimator -------------------------------
    #est_process.start()
    time.sleep(2)
    #---------------------------------Main COntroller----------------------------
    #controller_process.start()


    #------------------------join the processes to this main thread-------------------
    #controller_process.join()

    sensor_process.kill()
    #est_process.kill()

    print("Exit")
