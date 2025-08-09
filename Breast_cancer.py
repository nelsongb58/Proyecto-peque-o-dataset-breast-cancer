import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta para guardar imágenes si no existe
os.makedirs("images", exist_ok=True)

# 1. Cargar el dataset
df = pd.read_csv("breast_cancer_dataset.csv")

# 2. Vista general
print("Primeras filas del dataset:")
print(df.head())

print("Estadísticas descriptivas:")
print(df.describe())

print("Información del dataset:")
print(df.info())

# 3. Limpieza básica
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# 4. Verificar distribución de clases
print("📈 Distribución de diagnóstico:")
print(df["diagnosis"].value_counts())

# 5. Visualización de variables clave
sns.set(style="whitegrid")
features = ["radius_mean", "area_mean", "concavity_mean"]
plt.figure(figsize=(16, 5))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x="diagnosis", y=feature, data=df, palette=["#66c2a5", "#fc8d62"])
    plt.xticks([0, 1], ["Benigno", "Maligno"])
    plt.title(f"{feature} por diagnóstico")
plt.tight_layout()
plt.savefig("images/boxplot_variables_clave.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Comparación de medias por diagnóstico
mean_table = df.groupby("diagnosis").mean().T
mean_table.columns = ["Benigno", "Maligno"]
mean_table["Diferencia"] = abs(mean_table["Maligno"] - mean_table["Benigno"])
mean_table = mean_table.sort_values("Diferencia", ascending=False)
print("Comparación de medias por diagnóstico:")
print(mean_table.round(2))

# 7. Gráfico de barras: variables más discriminantes
top_features = mean_table.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features["Diferencia"], y=top_features.index, palette="Reds_r")
plt.title("Variables más discriminantes entre benignos y malignos")
plt.xlabel("Diferencia de medias")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig("images/top_variables_discriminantes.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Análisis por grupo de variables
mean_vars = [col for col in df.columns if "_mean" in col]
se_vars = [col for col in df.columns if "_se" in col]
worst_vars = [col for col in df.columns if "_worst" in col]

mean_diff = mean_table.loc[mean_vars]["Diferencia"].mean()
se_diff = mean_table.loc[se_vars]["Diferencia"].mean()
worst_diff = mean_table.loc[worst_vars]["Diferencia"].mean()

group_diff = pd.DataFrame({
    "Grupo": ["Mean", "SE", "Worst"],
    "Diferencia promedio": [mean_diff, se_diff, worst_diff]
})

# 9. Visualización de diferencias por grupo
plt.figure(figsize=(8, 5))
sns.barplot(x="Diferencia promedio", y="Grupo", data=group_diff, palette="Blues_r")
plt.title("Comparación de grupos de variables")
plt.xlabel("Diferencia promedio entre clases")
plt.ylabel("Grupo de variables")
plt.tight_layout()
plt.savefig("images/comparacion_grupos_variables.png", dpi=300, bbox_inches='tight')
plt.show()
