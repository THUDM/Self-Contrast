import os
import math
import json
import matplotlib.pyplot as plt

from brokenaxes import brokenaxes
from compute_winrate import get_winrate


data = {
    "Self-Contrast": {
        1: 0,
        2: 0,
        4: 0,
        8: 0,
        16: 0,
    },
    "$\mathregular{DPO_{std}}$": {
        1: 0,
        2: 0,
        4: 0,
        8: 0,
        16: 0,
    },
    "SFT": {
        1: 0,
        2: 0,
        4: 0,
        8: 0,
        16: 0,
    },
    "SPIN": {
        1: 0,
    }
}


result_path = "/workspace/xixuan/Self-Contrast/results/hh-rlhf/test/reward"
winrates = get_winrate(result_path)
for model_name, (winrate, average_reward) in winrates.items():
    for i in [1, 2, 4, 8, 16]:
        if model_name.endswith(f"Self-Contrast_{i}"):
            data['Self-Contrast'][i] = winrate
            break
        if "DPO" in model_name:
            data['$\mathregular{DPO_{std}}$'][i] = winrate
        if "SFT" in model_name:
            data['SFT'][i] = winrate
        if "SPIN" in model_name and i == 1:
            data['SPIN'][i] = winrate


plt.figure(figsize=(5.4 / 1.25, 4 / 1.25))

bax = brokenaxes(
    ylims=((int(100 * data['SFT'][1] - 1), int(100 * data['SFT'][1] + 1)), (77, 87)),
    hspace=0.1,
    despine=False,
    diag_color='black',
    d=0.010,
)

for key, value in data.items():
    if "SFT" in key:
        bax.plot(value.keys(), [v * 100 for v in value.values()], '--', label=key, zorder=1)
    elif "DPO" in key:
        bax.plot(value.keys(), [v * 100 for v in value.values()], '--', label=key, zorder=1)
    else:
        bax.plot(value.keys(), [v * 100 for v in value.values()], label=key, zorder=5, marker='o')


bax.grid(True, linestyle='--')
bax.set_xlabel('Negative Sample Quantity', labelpad=15.6)
bax.set_ylabel('Win Rate', labelpad=20)
plt.title('HH-RLHF')
bax.set_xscale('log', base=2)
bax.set_xticks([1, 2, 4 ,8, 16], [1, 2, 4 ,8 ,16])

bax.legend(ncol=1, loc='lower right', frameon=True)

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/HH-RLHF.png')
plt.show()

