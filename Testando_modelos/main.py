import json
import os
from pathlib import Path

import psutil
from llama_cpp import Llama
from Testando_modelos import download_modelos, download_modelos_gguf
from Testando_modelos.cenarios import cenario_A, cenario_B, cenario_C, cenario_D
from Testando_modelos.metricas import MetricasInferencia


def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")


def modelo_eliminado(modelo: str, formato: str = "safetensors") -> bool:
    nome_arquivo = modelo.replace("/", "--")

    if formato == "gguf":
        nome_arquivo += "-gguf"
        pasta = "resultados_gguf"
    else:
        pasta = "resultados"

    arquivo = Path(__file__).parent / pasta / f"{nome_arquivo}.json"

    if not arquivo.exists():
        return False

    with open(arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)

    return dados.get("eliminado", False) or dados.get("eliminacao_manual", {}).get("eliminado", False)


def escolher_modelo(lista_modelos: dict, rotulo: str) -> str | None:
    print(f"{rotulo}:")
    for id_modelo, modelo in lista_modelos.items():
        print(f"id {id_modelo} do {rotulo}: {modelo['Modelo']}")
    escolha = input(f"digite o id do {rotulo}: ").strip()
    if escolha not in lista_modelos:
        print(f"id '{escolha}' inválido — nenhum modelo com esse id.")
        return None
    return escolha


def testar_safetensors():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base_dir = Path(__file__).parent
    with open(base_dir / "lista_de_modelos.json", "r", encoding="utf-8") as arquivo:
        lista_modelos = json.load(arquivo)

    limpar_tela()
    x = escolher_modelo(lista_modelos, "modelo")
    if x is None:
        return

    modelo_nome = lista_modelos[x]["Modelo"]
    metricas = MetricasInferencia()
    nome_pasta = modelo_nome.replace("/", "--")
    caminho = base_dir / "modelos_local" / nome_pasta

    if not caminho.exists():
        print(f"\n⚠️  O modelo '{modelo_nome}' ainda não foi baixado.")
        print(f"    Esperado em: {caminho}")
        print(f"    Baixe primeiro com a opção 1 (baixar modelo), id {x}.\n")
        return

    print("Carregando modelo...")
    metricas.tempo_load_modelo_inicial()
    tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(caminho, local_files_only=True)
    model.eval()
    metricas.tempo_load_modelo_final()
    print(f"Pronto! o modelo: {modelo_nome} foi carregado com sucesso!")

    cenario_A.cenario_a(modelo_nome, tokenizer, model, metricas)
    if modelo_eliminado(modelo_nome):
        print(f"Modelo {modelo_nome} foi eliminado. Pulando cenários restantes.")
        return
    cenario_B.cenario_b(modelo_nome, tokenizer, model, metricas)
    if modelo_eliminado(modelo_nome):
        print(f"Modelo {modelo_nome} foi eliminado. Pulando cenários restantes.")
        return
    cenario_C.cenario_c(modelo_nome, tokenizer, model)
    if modelo_eliminado(modelo_nome):
        print(f"Modelo {modelo_nome} foi eliminado. Pulando cenários restantes.")
        return
    cenario_D.cenario_d(modelo_nome, tokenizer, model)
    if modelo_eliminado(modelo_nome):
        print(f"Modelo {modelo_nome} foi eliminado. Pulando cenários restantes.")
        return


def testar_gguf():
    base_dir = Path(__file__).parent
    with open(base_dir / "lista_de_modelos_gguf.json", "r", encoding="utf-8") as arquivo:
        lista_modelos = json.load(arquivo)

    limpar_tela()
    x = escolher_modelo(lista_modelos, "modelo_gguf")
    if x is None:
        return

    metricas = MetricasInferencia()
    modelo_id = lista_modelos[x]["Modelo"]
    arquivo_gguf = lista_modelos[x]["Arquivo"]
    nome_pasta = modelo_id.replace("/", "--")
    caminho_gguf = base_dir / "modelos_local_gguf" / nome_pasta / arquivo_gguf

    if not caminho_gguf.exists():
        print(f"\n⚠️  O modelo '{modelo_id}' ainda não foi baixado.")
        print(f"    Esperado em: {caminho_gguf}")
        print(f"    Baixe primeiro com a opção 2 (baixar modelo_gguf), id {x}.\n")
        return

    print("Carregando modelo_gguf...")
    metricas.tempo_load_modelo_inicial()
    processo = psutil.Process(os.getpid())
    print("Antes:", processo.memory_info().rss / (1024 ** 2), "MB")
    metricas.iniciar()

    llm = Llama(
        model_path=str(caminho_gguf),
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
    )

    metricas.tempo_load_modelo_final()
    print("Depois:", processo.memory_info().rss / (1024 ** 2), "MB")
    print(f"Pronto! modelo: {modelo_id} carregado em {metricas.tempo_carregamento_modelo:.2f}s")

    print("Iniciando cenario A: Cold Start ...")
    cenario_A.cenario_a_gguf(llm, modelo_id, metricas)
    if modelo_eliminado(modelo_id, formato="gguf"):
        print(f"Modelo {modelo_id} foi eliminado. Pulando cenários restantes.")
        return

    print("Iniciando cenario B: Geração em Série ...")
    cenario_B.cenario_b_gguf(llm, modelo_id, metricas)
    if modelo_eliminado(modelo_id, formato="gguf"):
        print(f"Modelo {modelo_id} foi eliminado. Pulando cenários restantes.")
        return

    print("Iniciando cenario C: Pressão de JSON em Português ...")
    cenario_C.cenario_c_gguf(llm, modelo_id)
    if modelo_eliminado(modelo_id, formato="gguf"):
        print(f"Modelo {modelo_id} foi eliminado. Pulando cenários restantes.")
        return

    print("Iniciando cenario D: Sem Contexto BNCC ...")
    cenario_D.cenario_d_gguf(llm, modelo_id)
    print("testes finalizado")


def main():
    limpar_tela()
    while True:
        print("O que vc quer fazer")
        print("0 - Sair")
        print("1 - baixar modelo")
        print("2 - baixar modelo_gguf")
        print("3 - testa modelo conventional")
        print("4 - testa modelo_gguf")
        resposta = input("digite sua resposta: ").strip()
        if resposta == "0":
            break
        elif resposta == "1":
            limpar_tela()
            download_modelos.main()
        elif resposta == "2":
            limpar_tela()
            download_modelos_gguf.download()
        elif resposta == "3":
            testar_safetensors()
        elif resposta == "4":
            testar_gguf()
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    main()

