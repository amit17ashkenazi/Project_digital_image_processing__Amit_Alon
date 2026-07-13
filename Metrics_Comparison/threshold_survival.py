import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = r"C:\Users\alonk\GitHub\Project_digital_image_processing__Amit_Alon\Metrics_Comparison\lane_comparison_results.csv"
MAX_ERROR = 45  # px
FIXED_LEVEL = 5  # pick one of the 9 levels for the bar chart

AUGMENTATIONS = ["motion_blur", "low_light", "rain"]
NUM_LEVELS = 9

df = pd.read_csv(INPUT_CSV)


def survival_rate(sub, max_error):
    total = 0
    survived = 0
    for _, r in sub.iterrows():
        for side in ["left", "right"]:
            diff = r[f"{side}_bottom_diff_px"]
            lost = r[f"{side}_lost"]
            if pd.isna(diff) and not lost:
                continue
            total += 1
            if not lost and not pd.isna(diff) and diff <= max_error:
                survived += 1
    return survived / total if total > 0 else np.nan


results = {}
for aug in AUGMENTATIONS:
    results[aug] = {}
    for level in range(1, NUM_LEVELS + 1):
        sub = df[(df["augmentation"] == aug) & (df["level"] == level)]
        rate = survival_rate(sub, MAX_ERROR)
        avg_snr = sub["achieved_snr_db"].mean()
        results[aug][level] = {"survival": rate, "snr": avg_snr}

# =========================
# Graph 1: Bar chart at a fixed level (unchanged)
# =========================
fig1, ax1 = plt.subplots(figsize=(7, 5))

values = [results[aug][FIXED_LEVEL]["survival"] for aug in AUGMENTATIONS]
snr_labels = [f"{aug}\n({results[aug][FIXED_LEVEL]['snr']:.1f} dB)" for aug in AUGMENTATIONS]

ax1.bar(snr_labels, values, color=["#4C72B0", "#DD8452", "#55A868"])
ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Clean baseline")

ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Qualitative Survival Rate")
ax1.set_title(f"Survival Rate by Augmentation (Level {FIXED_LEVEL})")
ax1.legend()
plt.tight_layout()
plt.savefig("survival_rate_bar_chart.png", dpi=150)
plt.show()

# =========================
# Graphs 2-4: One separate SNR line chart per augmentation,
# X-axis reversed so higher SNR (cleaner) appears first/left,
# lower SNR (noisier) appears last/right.
# =========================
colors = {"motion_blur": "#4C72B0", "low_light": "#DD8452", "rain": "#55A868"}

for aug in AUGMENTATIONS:
    snrs = [results[aug][level]["snr"] for level in range(1, NUM_LEVELS + 1)]
    rates = [results[aug][level]["survival"] for level in range(1, NUM_LEVELS + 1)]

    # Sort by SNR descending (high SNR first, low SNR last)
    order = np.argsort(snrs)[::-1]
    snrs_sorted = np.array(snrs)[order]
    rates_sorted = np.array(rates)[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(snrs_sorted, rates_sorted, marker="o", color=colors[aug], label=aug)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Clean baseline")

    ax.set_xlabel("Measured SNR (dB)")
    ax.set_ylabel("Qualitative Survival Rate")
    ax.set_title(f"Survival Rate vs Measured SNR — {aug}")
    ax.set_ylim(0, 1.1)
    ax.invert_xaxis()  # ensure high SNR (clean) on the left, low SNR (noisy) on the right
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"survival_rate_vs_snr_{aug}.png", dpi=150)
    plt.show()