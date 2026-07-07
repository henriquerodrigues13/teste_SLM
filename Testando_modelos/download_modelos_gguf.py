import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def download():
    BASE_DIR = Path(__file__).parent

    with open(BASE_DIR / "lista_de_modelos_gguf.json", "r", encoding="utf-8") as f:
        lista_modelos_gguf = json.load(f)

    print("Escolha um modelo GGUF para baixar...")
    for x in lista_modelos_gguf:
        print(f"id {x} do modelo: {lista_modelos_gguf[x]['Modelo']} ({lista_modelos_gguf[x]['Quantizacao']})")

    id_modelo = input("Digite o id do modelo: ")

    modelo = lista_modelos_gguf[id_modelo]["Modelo"]
    arquivo = lista_modelos_gguf[id_modelo]["Arquivo"]

    nome_pasta = modelo.replace("/", "--")
    pasta_destino = BASE_DIR / "modelos_local_gguf" / nome_pasta

    print(f"Baixando {arquivo} de {modelo}...")

    caminho = hf_hub_download(
        repo_id=modelo,
        filename=arquivo,
        local_dir=pasta_destino
    )

    print(f"Modelo baixado em: {caminho}")


