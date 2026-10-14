import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Generación de dataset con ruido sintético
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],  # 120 es atípico/erróneo
    'ingreso_mensual': [
        15000, 22000, 18000, np.nan, 12000,
        950000, 25000, 21000, 19000, np.nan
    ],  # 950000 es un outlier
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}

df_dirty = pd.DataFrame(data)

print("CONJUNTO DE DATOS ORIGINAL")
print(df_dirty)

# 1. Tratamiento de valores atípicos por IQR
# Se aplica sobre la variable ingreso_mensual
Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
print(f"Q1: ${Q1:,.2f}")
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)
print(f"Q3: ${Q3:,.2f}")

IQR = Q3 - Q1
print(f"IQR: ${IQR:,.2f}")
limite_superior = Q3 + 1.5 * IQR

print(f"\nLímite Superior para Outliers en Ingreso: ${limite_superior:,.2f}")

# Reemplazar outliers extremos con NaN
# para después tratarlos mediante imputación
df_dirty.loc[
    df_dirty['ingreso_mensual'] > limite_superior,
    'ingreso_mensual'
] = np.nan
print("\n=== VALORES ATÍPICOS")
print(df_dirty)

# Corrección de edad errónea
# Se consideran inválidas edades mayores a 100 años
df_dirty.loc[
    df_dirty['edad'] > 100,
    'edad'
] = np.nan

print("\n=== DATASET DESPUÉS DE ELIMINAR OUTLIERS ===")
print(df_dirty)

# 2. Imputación mediante KNN
# Se utilizan los 2 vecinos más cercanos
imputer = KNNImputer(n_neighbors=2)

df_clean_array = imputer.fit_transform(df_dirty)

# Convertir nuevamente el resultado en DataFrame
df_clean = pd.DataFrame(
    df_clean_array,
    columns=df_dirty.columns
)

print("\n=== DATASET LIMPIO E IMPUTADO ===")
print(df_clean)
