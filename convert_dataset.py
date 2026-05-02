import pandas as pd

# Convert each txt file to CSV
for split in ['train', 'test', 'val']:
    df = pd.read_csv(f"data/{split}.txt", sep=";", names=['text','emotion'])
    df.to_csv(f"data/{split}.csv", index=False)

# Combine all CSV files into one
combined = pd.concat([pd.read_csv(f"data/{s}.csv") for s in ['train', 'test', 'val']])
combined.to_csv("data/emotion_dataset.csv", index=False)
print("✅ Combined dataset saved as emotion_dataset.csv")
