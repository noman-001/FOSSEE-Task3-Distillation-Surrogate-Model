import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 1000  

feed_temp = np.random.uniform(25.0, 85.0, N_SAMPLES)          # °C
feed_pressure = np.random.uniform(1.0, 2.5, N_SAMPLES)        # atm
feed_x_benzene = np.random.uniform(0.3, 0.7, N_SAMPLES)       # Mole fraction Benzene
num_stages = np.random.randint(10, 20, N_SAMPLES)             # Stage count
feed_stage = np.round(num_stages * np.random.uniform(0.4, 0.6, N_SAMPLES)).astype(int) # Feed stage
reflux_ratio = np.random.uniform(1.2, 4.0, N_SAMPLES)         # Reflux ratio
bottoms_rate = np.random.uniform(8.0, 18.0, N_SAMPLES)        # kmol/h

xD_benzene = 0.82 + 0.12 * (1 - np.exp(-0.8 * reflux_ratio)) + 0.15 * feed_x_benzene - 0.005 * (bottoms_rate / 10.0)
xD_benzene = np.clip(xD_benzene + np.random.normal(0, 0.003, N_SAMPLES), 0.85, 0.998)

xB_toluene = 0.80 + 0.13 * (1 - np.exp(-0.7 * reflux_ratio)) + 0.12 * (1 - feed_x_benzene) + 0.003 * bottoms_rate
xB_toluene = np.clip(xB_toluene + np.random.normal(0, 0.003, N_SAMPLES), 0.82, 0.995)

Qc_kW = 250.0 + 110.0 * reflux_ratio + 12.0 * bottoms_rate + 0.8 * feed_temp + np.random.normal(0, 3, N_SAMPLES)

Qr_kW = Qc_kW + 40.0 + 1.5 * feed_pressure * 10 + np.random.normal(0, 2, N_SAMPLES)

df = pd.DataFrame({
    'Feed_Temp_C': np.round(feed_temp, 2),
    'Feed_Pressure_atm': np.round(feed_pressure, 2),
    'Feed_x_Benzene': np.round(feed_x_benzene, 4),
    'Num_Stages': num_stages,
    'Feed_Stage': feed_stage,
    'Reflux_Ratio': np.round(reflux_ratio, 4),
    'Bottoms_Rate_kmol_h': np.round(bottoms_rate, 2),
    'xD_Benzene': np.round(xD_benzene, 4),
    'xB_Toluene': np.round(xB_toluene, 4),
    'Qc_kW': np.round(Qc_kW, 2),
    'Qr_kW': np.round(Qr_kW, 2)
})

csv_path = "Dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Dataset generated successfully with {len(df)} samples across all required inputs!")