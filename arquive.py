# import h5py
# import pandas as pd
# import os

# pasta_raiz = 'data/'

# lista_dfs = []

# for root, dirs, files in os.walk(pasta_raiz):
#     for file in files:
#         if file.endswith('.h5'):
#             caminho_arquivo = os.path.join(root, file)
#             print(f"Lendo arquivo: {caminho_arquivo}")
            
#             with h5py.File(caminho_arquivo, 'r') as f:
#                 dados = f['radar_data'][:]
#                 df = pd.DataFrame(dados)
#                 lista_dfs.append(df)

# df_final = pd.concat(lista_dfs, ignore_index=True)
# df_final.to_csv('dados_concatenados.csv', index=False)

# print("Exportação finalizada em dados_concatenados.csv")

import pandas as pd

# Parâmetros
arquivo_origem = 'dados_concatenados.csv'
arquivo_destino = 'dados_concatenados_short.csv'
coluna_classe = 'label_id'
n_por_classe = 60000
chunksize = 500_000

coletadas = {}  # Dict: classe → [amostras]
total_por_classe = {}  # Dict: classe → número atual coletado

print(" Iniciando leitura em chunks...")
leitor = pd.read_csv(arquivo_origem, chunksize=chunksize)

for i, chunk in enumerate(leitor, 1):
    print(f"\n Processando chunk {i}...")

    for classe, grupo in chunk.groupby(coluna_classe):
        if classe not in coletadas:
            coletadas[classe] = []
            total_por_classe[classe] = 0

        faltando = n_por_classe - total_por_classe[classe]
        if faltando > 0:
            amostras = grupo.sample(min(faltando, len(grupo)), random_state=42)
            coletadas[classe].append(amostras)
            total_por_classe[classe] += len(amostras)

    # Mostrar progresso por classe
    print(" Progresso:")
    for c in sorted(coletadas):
        status = f"{total_por_classe[c]}/{n_por_classe}"
        completo = if total_por_classe[c] >= n_por_classe else "⏳"
        print(f"  Classe {c:>2}: {status} {completo}")

    # Verifica se já atingiu todas as classes
    completas = [total_por_classe[c] >= n_por_classe for c in coletadas]
    if len(completas) >= 11 and all(completas):
        print("\n Coleta completa para todas as 11 classes!")
        break

# Junta amostras e salva
df_final = pd.concat([pd.concat(v).head(n_por_classe) for v in coletadas.values()])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

df_final.to_csv(arquivo_destino, index=False)
print(f"\n Arquivo '{arquivo_destino}' salvo com {len(df_final)} linhas.")
