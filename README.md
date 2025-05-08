# 🐾 Unitree Go2 - RL Controller (Fast Forward Demo)

![Unitree Go2](https://unitree.com/wp-content/uploads/2023/03/go2_img01-1024x585.jpg)

This project showcases a **fast-forward version** of the **Reinforcement Learning (RL) controller/policy** deployed on a **Unitree Go2** quadruped robot.

Watch the demo on YouTube:
[![RL Controller Demo - Unitree Go2](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=8bq4slHcozQ)

## 📝 Project Overview

- 🎯 **Objective**: Learn a locomotion controller using RL for the Unitree Go2 and deploy it on real hardware.
- 🚀 **Deployment**: The trained policy is executed on the Go2 robot in real time.
- ⏩ **Fast-forwarded Playback**: The video demo is sped up to highlight gait stability, trajectory following, and body dynamics.
- ⚙️ **Frameworks**: Training was done using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), simulation via MuJoCo, and deployment integrated with the Unitree SDK.

## 📂 Repository Structure

```bash
📁 controller/          # Policy implementation and deployment interface
📁 training/            # RL training scripts and environment definitions
📁 scripts/             # Launch, logging, and post-processing tools
📄 README.md            # You're here!
