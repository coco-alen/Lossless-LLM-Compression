import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
epoch = [150, 200, 250, 300]

MobileNetV2 = [69.03, 69.59, 70.59, 70.22]
ax.plot(epoch, MobileNetV2, color=sns.color_palette()[0], alpha=1, linestyle='-', linewidth=2, marker='o', markersize="10" , label = 'MobileNetV2')

MobileNetV3 = [68.22,	68.45,	69.32,	69.84]
ax.plot(epoch, MobileNetV3, color=sns.color_palette()[2], alpha=1, linestyle='-', linewidth=2, marker='v', markersize="10" , label = 'MobileNetV3')

ProxylessNAS = [69.68,	70.45,	70.86,	70.71]
ax.plot(epoch, ProxylessNAS, color=sns.color_palette()[4], alpha=1, linestyle='-', linewidth=2, marker='h', markersize="10" , label = 'ProxylessNAS')

MCUNet = [71.81,	71.63,	71.38,	71.41]
ax.plot(epoch, MCUNet, color=sns.color_palette()[6], alpha=1, linestyle='-', linewidth=2, marker='D', markersize="10" , label = 'MCUNet')

MobileNetV2_Tiny = [68.05,	69.13,	69.3	,69.78]
ax.plot(epoch, MobileNetV2_Tiny, color=sns.color_palette()[3], alpha=1, linestyle='-', linewidth=2, marker='*', markersize="18" , label = "MobileNetV2-Tiny")

ax.set_xlabel('Epoch', size=12)
ax.set_ylabel('Accuarcy', size=12)
# ax1.tick_params('y', colors='r')
ax.grid(True)

# Save the plot as PDF
plt.legend(loc='upper left', prop={'size': 12})
plt.tight_layout()
plt.savefig("./ablation_epoch.pdf")
plt.show()
