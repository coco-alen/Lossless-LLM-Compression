import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(3, 3))

deepShift_energy = [0.536, 0.74, 1.736, 8.361]
deepShift_accy = [68.42, 69.45, 71.39, 73.92]
ax.plot(deepShift_energy, deepShift_accy, color=sns.color_palette()[0], alpha=1, linestyle='-', linewidth=2, marker='o', markersize="10" , label = 'Deepshift')

adderNet_energy = [0.697, 1.091, 1.281, 3.038, 16]
adderNet_accy = [66.57, 67.85, 68.87, 69.02, 72.64]
ax.plot(adderNet_energy, adderNet_accy, color=sns.color_palette()[2], alpha=1, linestyle='-', linewidth=2, marker='v', markersize="10" , label = 'AdderNet')

# shiftaddNas_energy = [1.0, 3.972]
# shiftaddNas_accy = [71, 78.6]
# ax.plot(shiftaddNas_energy, shiftaddNas_accy, color=sns.color_palette()[5], alpha=1, linestyle='-', linewidth=2, marker='s', markersize="10" , label = 'ShiftAddNas')

apot_energy = [0.386, 0.521]
apot_accy = [69.51, 70.12]
ax.plot(apot_energy, apot_accy, color=sns.color_palette()[4], alpha=1, linestyle='-', linewidth=2, marker='h', markersize="10" , label = 'APoT')

mult_energy = [1.726, 2.345, 2.471, 4.28]
mult_accy = [69.32, 70.59, 70.86, 71.38]
ax.plot(mult_energy, mult_accy, color=sns.color_palette()[6], alpha=1, linestyle='-', linewidth=2, marker='D', markersize="10" , label = 'Mult.')

shiftaddaug_energy = [0.697, 0.74, 0.774, 1.323]
shiftaddaug_accy = [71.89, 71.83, 73.86, 74.59]
ax.plot(shiftaddaug_energy, shiftaddaug_accy, color=sns.color_palette()[3], alpha=1, linestyle='-', linewidth=2, marker='*', markersize="18" , label = r'$\bf (Ours)$')

# ax.set_xlabel('Energy', size=12)
# ax.set_ylabel('Accuarcy', size=12)
ax.set_xscale("log") # y 轴上以4为底数呈对数显示，2、3表示会标记出2倍、3倍的位置
# ax1.tick_params('y', colors='r')
ax.grid(True)

# Save the plot as PDF
plt.legend(loc='lower right', prop={'size': 8})
plt.tight_layout()
plt.savefig("./cifar100.png")
plt.show()
