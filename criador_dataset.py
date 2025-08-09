import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# --- PARÂMETROS DE CONFIGURAÇÃO ---
NUM_REBOCADORES = 80
DATA_INICIO = datetime(2024, 1, 1)
DIAS_SIMULACAO = 365
DIRETORIO_SAIDA = "dataset"
STRESS_THRESHOLD = 20 # Limite de pontos de estresse para a falha acontecer

# ... (Restante dos parâmetros permanece o mesmo) ...
PORTOS = {
    "Santos": {"lat": -23.98, "lon": -46.30},
    "Rio de Janeiro": {"lat": -22.89, "lon":-43.18},
    "Salvador": {"lat": -12.96, "lon": -38.51},
    "Rio Grande": {"lat": -32.09, "lon": -52.11},
    "Suape": {"lat": -8.39, "lon":-34.96}
}
ZONA_RISCO_SANTOS = {"lat_min": -24.05, "lat_max": -23.95, "lon_min": -46.35, "lon_max": -46.25}
ZONA_RISCO_RIO_GRANDE = {"lat_min": -32.15, "lat_max": -32.05, "lon_min": -52.15, "lon_max": -52.05}
SEMANAS_CHUVOSAS = [6, 7, 18, 19, 45, 46]

def criar_diretorio_se_nao_existir(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
        print(f"Diretório '{diretorio}' criado com sucesso.")

def esta_na_zona(lat, lon, zona):
    return zona['lat_min'] <= lat <= zona['lat_max'] and zona['lon_min'] <= lon <= zona['lon_max']

def gerar_dados_climaticos():
    # Esta função permanece a mesma da versão anterior
    print("\nGerando dados climáticos diários...")
    clima_records = []
    data_atual = DATA_INICIO
    for _ in range(DIAS_SIMULACAO):
        semana_do_ano = data_atual.isocalendar()[1]
        em_semana_chuvosa = semana_do_ano in SEMANAS_CHUVOSAS
        for porto in PORTOS.keys():
            if em_semana_chuvosa and random.random() < 0.75:
                condicao = random.choice(['Chuvoso', 'Tempestade'])
                umidade = random.randint(85, 98); temp = random.uniform(18.0, 25.0); vento = random.uniform(15.0, 40.0)
            else:
                condicao = random.choice(['Ensolarado', 'Nublado']); umidade = random.randint(60, 84); temp = random.uniform(22.0, 32.0); vento = random.uniform(5.0, 15.0)
            clima_records.append({"data": data_atual.date(), "porto": porto, "condicao_tempo": condicao, "temperatura_celsius": round(temp, 1), "umidade_percentual": umidade, "velocidade_vento_kmh": round(vento, 1)})
        data_atual += timedelta(days=1)
    df_clima = pd.DataFrame(clima_records)
    df_clima.to_csv(os.path.join(DIRETORIO_SAIDA, "clima_diario.csv"), index=False)
    print("-> clima_diario.csv gerado.")
    return df_clima


def gerar_dados_operacionais_obvios(df_rebocadores, df_clima):
    """Gera dados com regras de falha claras e determinísticas."""
    print("\nIniciando simulação...")
    
    operacoes_records = []
    manutencoes_records = []
    manutencao_id_counter = 1
    
    df_clima['data'] = pd.to_datetime(df_clima['data'])
    
    estado_rebocadores = {
        row['id_rebocador']: {
            "lat": PORTOS[row['porto_base']]['lat'], "lon": PORTOS[row['porto_base']]['lon'],
            "horas_desde_manutencao": 0, "porto_base": row['porto_base'],
            "stress_level": 0, "causa_stress": "Nenhuma"  # NOVO: Acumulador de estresse
        } for index, row in df_rebocadores.iterrows()
    }

    data_atual = DATA_INICIO
    for dia in range(DIAS_SIMULACAO):
        if (dia + 1) % 30 == 0:
            print(f"Processando simulação... Dia {dia+1}/{DIAS_SIMULACAO}")
        
        clima_do_dia = df_clima[df_clima['data'].dt.date == data_atual.date()].set_index('porto')

        for hora in range(24):
            timestamp_atual = data_atual + timedelta(hours=hora)
            for reb_id, estado in estado_rebocadores.items():
                estado['lat'] += np.random.randn() * 0.005
                estado['lon'] += np.random.randn() * 0.005
                
                # Valores base dos sensores
                temp_motor = random.uniform(75.0, 85.0)
                nivel_vibracao = random.uniform(10.0, 20.0)
                
                # --- LÓGICA DE ESTRESSE E SENSORES EXAGERADOS ---
                em_zona_santos = esta_na_zona(estado['lat'], estado['lon'], ZONA_RISCO_SANTOS)
                condicao_climatica_porto = clima_do_dia.loc[estado['porto_base']]['condicao_tempo']
                em_clima_adverso = condicao_climatica_porto in ['Chuvoso', 'Tempestade']

                if em_zona_santos:
                    temp_motor += 20  # Aumento óbvio
                    nivel_vibracao += 15 # Aumento óbvio
                    estado['stress_level'] += 1
                    estado['causa_stress'] = "Zona de Risco"
                elif em_clima_adverso:
                    temp_motor += 5
                    nivel_vibracao += 10
                    estado['stress_level'] += 1
                    estado['causa_stress'] = "Clima Adverso"
                else:
                    estado['causa_stress'] = "Nenhuma"
                
                operacoes_records.append({
                    "id_operacao": len(operacoes_records) + 1, "id_rebocador": reb_id, "timestamp": timestamp_atual,
                    "latitude": round(estado['lat'], 6), "longitude": round(estado['lon'], 6),
                    "temp_motor_celsius": round(temp_motor, 2), "nivel_vibracao_hz": round(nivel_vibracao, 2),
                })
                
                estado['horas_desde_manutencao'] += 1

                # --- LÓGICA DE FALHA DETERMINÍSTICA ---
                if estado['stress_level'] >= STRESS_THRESHOLD:
                    # Determina o componente baseado na causa do estresse
                    if estado['causa_stress'] == "Zona de Risco":
                        componente_falha = 'Sistema de Propulsão'
                    else: # Clima Adverso
                        componente_falha = 'Sistema Elétrico'
                    
                    manutencoes_records.append({
                        "id_manutencao": manutencao_id_counter, "id_rebocador": reb_id, "data_manutencao": timestamp_atual.date(),
                        "tipo_manutencao": "Corretiva Nao Planejada", "componente_afetado": componente_falha,
                        "custo_reparo_reais": random.randint(25000, 90000), "dias_inativo": random.randint(5, 12)
                    })
                    manutencao_id_counter += 1
                    estado['horas_desde_manutencao'] = 0
                    estado['stress_level'] = 0 # Zera o estresse após a falha
                
                elif estado['horas_desde_manutencao'] >= 720: # Manutenção Preventiva
                    manutencoes_records.append({
                        "id_manutencao": manutencao_id_counter, "id_rebocador": reb_id, "data_manutencao": timestamp_atual.date(),
                        "tipo_manutencao": "Preventiva Programada", "componente_afetado": "Revisao Geral",
                        "custo_reparo_reais": random.randint(5000, 12000), "dias_inativo": 1
                    })
                    manutencao_id_counter += 1
                    estado['horas_desde_manutencao'] = 0
                    estado['stress_level'] = 0 # Zera o estresse na preventiva também

        data_atual += timedelta(days=1)
    
    print("\nFinalizando e salvando arquivos CSV...")
    pd.DataFrame(operacoes_records).to_csv(os.path.join(DIRETORIO_SAIDA, "operacoes.csv"), index=False)
    print("-> operacoes.csv gerado.")
    pd.DataFrame(manutencoes_records).to_csv(os.path.join(DIRETORIO_SAIDA, "manutencoes.csv"), index=False)
    print("-> manutencoes.csv gerado.")


def main():
    criar_diretorio_se_nao_existir(DIRETORIO_SAIDA)
    # Gerar Rebocadores
    print("Gerando dados da frota de rebocadores...")
    rebocadores_data = [{"id_rebocador": i, "nome_rebocador": f"WS-{random.choice(list(PORTOS.keys()))[:3].upper()}-{i:02d}", "porto_base": random.choice(list(PORTOS.keys())), "ano_fabricacao": random.randint(2005, 2023)} for i in range(1, NUM_REBOCADORES + 1)]
    df_rebocadores = pd.DataFrame(rebocadores_data)
    df_rebocadores.to_csv(os.path.join(DIRETORIO_SAIDA, "rebocadores.csv"), index=False)
    print("-> rebocadores.csv gerado.")
    df_clima = gerar_dados_climaticos()
    gerar_dados_operacionais_obvios(df_rebocadores, df_clima)
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()
