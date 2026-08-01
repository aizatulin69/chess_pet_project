import pandas as pd
import matplotlib.pyplot as plt
import re

with open('pretrain_stats.csv', 'r') as f:
    content = f.read()

pattern = r'(\d+),"(\[.*?\])"'
matches = re.findall(pattern, content, re.DOTALL)

all_data = []
for epoch_str, data_str in matches:
    epoch = int(epoch_str)
    data_str = data_str.replace("'", '"')
    data_list = eval(data_str)
    for i, record in enumerate(data_list):
        record['epoch'] = epoch
        record['step'] = i
        all_data.append(record)

df = pd.DataFrame(all_data)
for col in ['ploss', 'vloss', 'accuracy', 'value']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['global_step'] = df['epoch'] * (df.groupby('epoch')['step'].transform('max') + 1) + df['step']

# ====== УБИРАЕМ СПАЙКИ ======
df['ploss_clean'] = df['ploss'].where(df['ploss'] < 4.5)
df['ploss_clean'] = df['ploss_clean'].interpolate(method='linear')

df['vloss_clean'] = df['vloss'].where(df['vloss'] < 1)
df['vloss_clean'] = df['vloss_clean'].interpolate(method='linear')

df['value_clean'] = df['value'].where(df['value'] > -0.2)
df['value_clean'] = df['value_clean'].interpolate(method='linear')
# ============================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Metrics (All 10 Epochs)', fontsize=16, fontweight='bold')

metrics = ['ploss_clean', 'vloss_clean', 'accuracy', 'value_clean']
original_metrics = ['ploss', 'vloss', 'accuracy', 'value_clean']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
titles = ['Policy Loss', 'Value Loss', 'Accuracy', 'Value Prediction']

for ax, metric, orig, color, title in zip(axes.flat, metrics, original_metrics, colors, titles):
    ax.plot(df['global_step'], df[metric], color=color, linewidth=0.3, alpha=0.9)
    window = 1000
    rolling = df[metric].rolling(window=window, min_periods=1).mean()
    ax.plot(df['global_step'], rolling, color='black', linewidth=1.5, alpha=0.8, label=f'MA({window})')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Global Step')
    ax.set_ylabel(orig)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Эпох: {df['epoch'].nunique()}")
print(f"Всего шагов: {len(df)}")
print(f"\nAccuracy по эпохам (max):")
print(df.groupby('epoch')['accuracy'].max())
print(f"\nPolicy Loss по эпохам (min):")
print(df.groupby('epoch')['ploss'].min())
print(f"\nValue Loss по эпохам (min):")
print(df.groupby('epoch')['vloss'].min())
