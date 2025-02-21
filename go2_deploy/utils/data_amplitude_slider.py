import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import inter_process_com.publisher as pub

# Create a figure and a set of subplots
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)

# Initial joint positions
init_pos = [0, 1.57, 0.5175, -0.9879, 0.4704,
            -1.57, 1.57, 0.5175, -0.9879, 0.4704]  

# Sliders
sliders = []
ax_sliders = []

#PJ = pub.clk_cmd()
slider_range = {
    'hj_FL' :np.array([-1, 1, 0]),
    'tj_FL' :np.array([-1.5, 3.3, 3]),
    'cj_FL' : np.array([-2.5, -0.9, -1]) 
}
i = 0

JC = pub.publish_joint_cmd()


for key, values in slider_range.items():

    ax_slider = plt.axes([0.25, 0.35 - 0.03 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, key, values[0], values[1], valinit=values[2])
    ax_sliders.append(ax_slider)
    sliders.append(slider)
    i += 1

def update(val):
    joint_positions = [slider.val for slider in sliders]
    print(f"Updated joint positions: {joint_positions}")
    # Replace with your publish command
    clock = np.zeros(2)
    JC.set(np.array(joint_positions[0]))

for slider in sliders:
    slider.on_changed(update)

plt.show()
