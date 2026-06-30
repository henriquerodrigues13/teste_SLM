import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from Testando_modelos import download_modelos
from Testando_modelos.cenarios import cenario_A, cenario_B, cenario_C
from Testando_modelos.metricas import MetricasInferencia


def modelo_eliminado(modelo: str) -> bool:
    nome_arquivo = modelo.replace("/", "--")
    arquivo = Path(__file__).parent / "resultados" / f"{nome_arquivo}.json"

    if not arquivo.exists():
        return False

    with open(arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)

    return dados.get("eliminado", False) or dados.get("eliminacao_manual", {}).get("eliminado", False)
while True:
    os.system("cls")
    print("O que vc quer fazer")
    print("0 - Sair")
    print("1 - baixar modelo")
    print("2 - testa modelo")
    reposta = str(input("digite sua reposta: "))
    if reposta == "0":
        break
    elif reposta == "1":
        os.system("cls")
        download_modelos.main()
    elif reposta == "2":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        caminho_json = os.path.join(BASE_DIR, 'lista_de_modelos.json')

        with open(caminho_json, 'r', encoding='utf-8') as arquivo:
            lista_modelos = json.load(arquivo)

        os.system("cls")
        print("modelo:")
        for id, modelo in lista_modelos.items():
            print(f"id {id} do modelo: {modelo['Modelo']}")
        x = input("digite o id do modelo: ")
        print("Carregando modelo...")

        metricas = MetricasInferencia()
        BASE_DIR = Path(__file__).parent
        nome_pasta = f"{lista_modelos[x]["Modelo"]}".replace("/", "--")
        caminho = BASE_DIR / "modelos_local" / nome_pasta

        metricas.tempo_load_modelo_inicial()
        tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(caminho, local_files_only=True)
        model.eval()
        metricas.tempo_load_modelo_final()
        print(f"Pronto! o modelo: {lista_modelos[x]["Modelo"]} foi carregado com sucesso!")
        cenario_A.cenario_a(lista_modelos[x]['Modelo'], tokenizer, model, metricas)
        if modelo_eliminado(lista_modelos[x]['Modelo']):
            print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            continue
        cenario_B.cenario_b(lista_modelos[x]['Modelo'], tokenizer, model, metricas)
        if modelo_eliminado(lista_modelos[x]['Modelo']):
            print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            continue
        cenario_C.cenario_c(lista_modelos[x]['Modelo'], tokenizer, model)
        if modelo_eliminado(lista_modelos[x]['Modelo']):
            print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            continue
