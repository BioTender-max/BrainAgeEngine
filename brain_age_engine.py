import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import shutil, os

# ── Parameters ───────────────────────────────────────────────────────────────
N = 500
ages = np.random.uniform(20, 90, N)

# ── Multi-modal features ─────────────────────────────────────────────────────
# Cortical thickness (mm): decreases with age
ct = 2.8 - 0.008*ages + np.random.normal(0, 0.15, N)
# White matter FA: decreases with age
fa = 0.55 - 0.003*ages + np.random.normal(0, 0.05, N)
# Functional connectivity: decreases with age
fc = 0.7 - 0.004*ages + np.random.normal(0, 0.08, N)
# DNA methylation clock (Horvath): increases with age
meth = ages * 0.95 + np.random.normal(0, 3, N)

# ── Brain age prediction (linear combination) ────────────────────────────────
feature_weights = np.array([-8.0, -15.0, -10.0, 0.9])  # CT, FA, FC, Meth
features = np.column_stack([ct, fa, fc, meth])
features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
brain_age_raw = (features_norm * feature_weights).sum(axis=1)
# Scale to age range
brain_age = ages.mean() + brain_age_raw * (ages.std() / (brain_age_raw.std() + 1e-9))
brain_age += np.random.normal(0, 3, N)  # residual noise

# ── Brain age gap ────────────────────────────────────────────────────────────
bag = brain_age - ages  # Brain Age Gap

# ── Cognitive decline correlation ────────────────────────────────────────────
# MMSE-like score: decreases with age and positive BAG
mmse = 30 - 0.05*ages - 0.3*bag + np.random.normal(0, 1.5, N)
mmse = np.clip(mmse, 0, 30)
r_bag_mmse, p_bag_mmse = pearsonr(bag, mmse)

# ── Neurodegeneration clock (AD/PD risk) ─────────────────────────────────────
ad_risk = 1 / (1 + np.exp(-(bag - 5) / 3))  # logistic
pd_risk = 1 / (1 + np.exp(-(bag - 3) / 4))

# ── Epigenetic reprogramming score ───────────────────────────────────────────
epi_score = meth - ages  # deviation from expected methylation age
epi_reprog = np.clip(-epi_score / 10, 0, 1)  # reprogramming potential

# ── Aging trajectory clustering (k-means style) ──────────────────────────────
n_clusters = 3
# Simple clustering by BAG tertiles
bag_tertiles = np.percentile(bag, [33, 67])
cluster_labels = np.digitize(bag, bag_tertiles)
cluster_names = ['Healthy Agers','Normal Agers','Accelerated Agers']
cluster_colors = ['#3fb950','#58a6ff','#f78166']

# ── Feature importance (correlation with brain age) ──────────────────────────
feat_names = ['Cortical Thickness','White Matter FA','Func. Connectivity','DNA Methylation']
feat_importance = np.abs([pearsonr(f, brain_age)[0] for f in [ct, fa, fc, meth]])

# ── Population stratification ────────────────────────────────────────────────
age_groups = ['20-40','40-60','60-90']
age_masks = [(ages < 40), (ages >= 40) & (ages < 60), (ages >= 60)]
group_bag = [bag[m] for m in age_masks]

# ── Key results ──────────────────────────────────────────────────────────────
r_pred, _ = pearsonr(ages, brain_age)
mae = np.abs(bag).mean()
mean_bag = bag.mean()
accel_agers = np.sum(cluster_labels == 2)

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Brain Age Engine — Multi-Modal Brain Age Prediction Dashboard',
             color='white', fontsize=16, fontweight='bold', y=0.98)

COLORS = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364','#ff7b72']

# Panel 1: Brain age prediction scatter
ax = axes[0, 0]; ax.set_facecolor('#161b22')
sc = ax.scatter(ages, brain_age, c=bag, cmap='RdYlGn_r', s=15, alpha=0.6)
cb = plt.colorbar(sc, ax=ax, label='Brain Age Gap')
cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
ax.plot([20, 90], [20, 90], 'w--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Chronological Age', color='white'); ax.set_ylabel('Predicted Brain Age', color='white')
ax.set_title(f'Brain Age Prediction (r={r_pred:.3f})', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 2: Age gap distribution
ax = axes[0, 1]; ax.set_facecolor('#161b22')
ax.hist(bag, bins=40, color='#58a6ff', alpha=0.8, edgecolor='#30363d')
ax.axvline(0, color='white', ls='--', lw=1.5, label='No gap')
ax.axvline(bag.mean(), color='#ffa657', ls='--', lw=2, label=f'Mean={bag.mean():.1f}y')
ax.set_xlabel('Brain Age Gap (years)', color='white'); ax.set_ylabel('Count', color='white')
ax.set_title('Brain Age Gap Distribution', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 3: Cognitive correlation
ax = axes[0, 2]; ax.set_facecolor('#161b22')
ax.scatter(bag, mmse, c=ages, cmap='plasma', s=15, alpha=0.6)
m, b = np.polyfit(bag, mmse, 1)
x_line = np.linspace(bag.min(), bag.max(), 100)
ax.plot(x_line, m*x_line+b, color='#ffa657', lw=2, label=f'r={r_bag_mmse:.3f}')
ax.set_xlabel('Brain Age Gap (years)', color='white'); ax.set_ylabel('MMSE Score', color='white')
ax.set_title('Cognitive Decline Correlation', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 4: Neurodegeneration risk
ax = axes[1, 0]; ax.set_facecolor('#161b22')
sorted_idx = np.argsort(bag)
ax.plot(bag[sorted_idx], ad_risk[sorted_idx], color='#f78166', lw=2, label='AD Risk')
ax.plot(bag[sorted_idx], pd_risk[sorted_idx], color='#d2a8ff', lw=2, label='PD Risk')
ax.axvline(0, color='white', ls='--', lw=1)
ax.set_xlabel('Brain Age Gap (years)', color='white'); ax.set_ylabel('Risk Probability', color='white')
ax.set_title('Neurodegeneration Risk (AD/PD)', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 5: Epigenetic clock
ax = axes[1, 1]; ax.set_facecolor('#161b22')
ax.scatter(ages, meth, c=epi_reprog, cmap='viridis', s=15, alpha=0.6)
ax.plot([20, 90], [20*0.95, 90*0.95], color='#ffa657', lw=2, ls='--', label='Expected')
ax.set_xlabel('Chronological Age', color='white'); ax.set_ylabel('Methylation Age (Horvath)', color='white')
ax.set_title('Epigenetic Clock', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 6: Aging trajectory
ax = axes[1, 2]; ax.set_facecolor('#161b22')
for cl, (name, col) in enumerate(zip(cluster_names, cluster_colors)):
    mask = cluster_labels == cl
    ax.scatter(ages[mask], brain_age[mask], color=col, s=15, alpha=0.6, label=f'{name} (n={mask.sum()})')
ax.plot([20, 90], [20, 90], 'w--', lw=1.5)
ax.set_xlabel('Chronological Age', color='white'); ax.set_ylabel('Brain Age', color='white')
ax.set_title('Aging Trajectory Clustering', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=7)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 7: Feature importance
ax = axes[2, 0]; ax.set_facecolor('#161b22')
sorted_fi = np.argsort(feat_importance)
ax.barh(range(len(feat_names)), feat_importance[sorted_fi],
        color=[COLORS[i] for i in sorted_fi], alpha=0.85, edgecolor='#30363d')
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels([feat_names[i] for i in sorted_fi], color='white', fontsize=9)
ax.set_xlabel('|Pearson r| with Brain Age', color='white')
ax.set_title('Feature Importance', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 8: Population stratification
ax = axes[2, 1]; ax.set_facecolor('#161b22')
bp = ax.boxplot(group_bag, patch_artist=True, medianprops=dict(color='white', lw=2))
for patch, color in zip(bp['boxes'], ['#3fb950','#58a6ff','#f78166']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels(age_groups, color='white')
ax.axhline(0, color='white', ls='--', lw=1)
ax.set_ylabel('Brain Age Gap (years)', color='white')
ax.set_title('BAG by Age Group', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 9: Summary
ax = axes[2, 2]; ax.set_facecolor('#161b22'); ax.axis('off')
ax.set_title('Summary Statistics', color='white', fontweight='bold')
summary_lines = [
    ('Subjects (N)', f'{N}'),
    ('Age Range', '20-90 years'),
    ('Prediction r', f'{r_pred:.3f}'),
    ('Mean Abs Error', f'{mae:.1f} years'),
    ('Mean BAG', f'{mean_bag:.1f} years'),
    ('BAG-MMSE r', f'{r_bag_mmse:.3f}'),
    ('Accel. Agers', f'{accel_agers} ({accel_agers/N*100:.1f}%)'),
    ('Top Feature', f'{feat_names[np.argmax(feat_importance)]}'),
    ('Modalities', '4 (CT, FA, FC, Meth)'),
]
for idx, (k, v) in enumerate(summary_lines):
    ax.text(0.05, 0.88 - idx*0.10, k, color='#8b949e', fontsize=10, transform=ax.transAxes)
    ax.text(0.65, 0.88 - idx*0.10, v, color='#58a6ff', fontsize=10, fontweight='bold', transform=ax.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/brain_age_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
shutil.copy(__file__, '/mnt/shared-workspace/shared/brain_age_engine.py')

print("=== BrainAgeEngine Key Results ===")
print(f"Subjects: {N}, Age range: 20-90 years")
print(f"Brain age prediction r: {r_pred:.3f}")
print(f"Mean absolute error: {mae:.1f} years")
print(f"Mean brain age gap: {mean_bag:.1f} years")
print(f"BAG-MMSE correlation: r={r_bag_mmse:.3f}")
print(f"Accelerated agers: {accel_agers} ({accel_agers/N*100:.1f}%)")
print(f"Top feature: {feat_names[np.argmax(feat_importance)]} (r={feat_importance.max():.3f})")
print("Dashboard saved.")
