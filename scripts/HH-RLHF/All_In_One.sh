# training
bash scripts/HH-RLHF/SFT.sh
bash scripts/HH-RLHF/SPIN.sh
bash scripts/HH-RLHF/DPO.sh
bash scripts/HH-RLHF/Self-Contrast_1.sh
bash scripts/HH-RLHF/Self-Contrast_2.sh
bash scripts/HH-RLHF/Self-Contrast_4.sh
bash scripts/HH-RLHF/Self-Contrast_8.sh
bash scripts/HH-RLHF/Self-Contrast_16.sh

# test
bash scripts/HH-RLHF/Test.sh

# show result
python src/compute_winrate.py

# plot
python src/draw_figures.py
