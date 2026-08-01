import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from ast import literal_eval

df = pd.read_csv("stats_150000.csv")
print(df.dtypes)

# Parse endings column
df['endings_parsed'] = df['endings'].apply(literal_eval)

# Create 2x2 grid for 4 metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Метрики обучения по количеству игр', fontsize=14)

metrics = [
    ('time', 'Время (time)', 'tab:blue'),
    ('policy_loss', 'Policy Loss', 'tab:orange'),
    ('value_loss', 'Value Loss', 'tab:green'),
    ('entropy', 'Entropy', 'tab:red')
]

for ax, (col, title, color) in zip(axes.flatten(), metrics):
    ax.plot(df['games'], df[col], marker='', color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Games')
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('metrics.png')
plt.show()
plt.close()

# Plot endings
endings_matrix = array(df['endings_parsed'].tolist())
num_categories = endings_matrix.shape[1]

endind_types = ["Истечение ходов", "Победа белых", "Победа чёрных", "Ничья", "Пат"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, e in enumerate(endind_types):
    ax.plot(df['games'], endings_matrix[:, i], marker='', label=f'{e}')

ax.set_title('Динамика концовок игр (Endings)')
ax.set_xlabel('Games')
ax.set_ylabel('Количество')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('endings.png')
plt.show()
plt.close()

print("Plots successfully created!")