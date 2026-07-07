import json
import os
from pathlib import Path
from time import sleep
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from Testando_modelos import download_modelos, download_modelos_gguf
from Testando_modelos.cenarios import cenario_A, cenario_B, cenario_C, cenario_D
from Testando_modelos.metricas import MetricasInferencia


def modelo_eliminado(modelo: str, formato: str = "safetensors") -> bool:
    nome_arquivo = modelo.replace("/", "--")

    if formato == "gguf":
        nome_arquivo += "-gguf"

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
    print("2 - baixar modelo_gguf")
    print("3 - testa modelo conventional")
    print("4 - testa modelo_gguf")
    reposta = str(input("digite sua reposta: "))
    if reposta == "0":
        break
    elif reposta == "1":
        os.system("cls")
        download_modelos.main()
    elif reposta == "2":
        os.system("cls")
        download_modelos_gguf.download()
    elif reposta == "3":
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
        cenario_D.cenario_d(lista_modelos[x]['Modelo'], tokenizer, model)
        if modelo_eliminado(lista_modelos[x]['Modelo']):
            print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            continue
    elif reposta == "4":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        caminho_json = os.path.join(BASE_DIR, 'lista_de_modelos_gguf.json')

        with open(caminho_json, 'r', encoding='utf-8') as arquivo:
            lista_modelos = json.load(arquivo)

        os.system("cls")
        metricas = MetricasInferencia()
        print("modelo_gguf:")
        for id, modelo in lista_modelos.items():
            print(f"id {id} do modelo_gguf: {modelo['Modelo']}")
        x = input("digite o id do modelo_gguf: ")

        modelo_id = lista_modelos[x]["Modelo"]
        arquivo_gguf = lista_modelos[x]["Arquivo"]
        nome_pasta = modelo_id.replace("/", "--")
        caminho_gguf = Path(__file__).parent / "modelos_local_gguf" / nome_pasta / arquivo_gguf

        print("Carregando modelo_gguf...")
        metricas.tempo_load_modelo_inicial()

        llm = Llama(
            model_path=str(caminho_gguf),
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
        )

        metricas.tempo_load_modelo_final()
        print(f"Pronto! modelo: {modelo_id} carregado em {metricas.tempo_carregamento_modelo:.2f}s")
        #cenario_A.cenario_a_gguf(llm, modelo_id, metricas)
        #if modelo_eliminado(lista_modelos[x]['Modelo'], formato="gguf"):
            #print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            #continue
        cenario_B.cenario_b_gguf(llm, modelo_id, metricas)
        if modelo_eliminado(lista_modelos[x]['Modelo'], formato="gguf"):
            print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            continue
        #cenario_C.cenario_c_gguf(llm, modelo_id, metricas)
        #if modelo_eliminado(lista_modelos[x]['Modelo'], formato="gguf"):
            #print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            #continue
        #cenario_D.cenario_d_gguf(llm, modelo_id, metricas)
        #if modelo_eliminado(lista_modelos[x]['Modelo'], formato="gguf"):
            #print(f"Modelo {lista_modelos[x]['Modelo']} foi eliminado. Pulando cenários restantes.")
            #continue

