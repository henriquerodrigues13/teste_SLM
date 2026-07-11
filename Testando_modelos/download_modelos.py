import os
import json

def baixando_modelo(modelo: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    try:
        print(f"baixando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(modelo)
        print(f"baixando modelo: {modelo}...")
        model = AutoModelForCausalLM.from_pretrained(modelo)
    except Exception as e:
        print(f"Erro: {e}")
    else:
        nome_pasta = modelo.replace("/", "--")
        caminho_salvar = os.path.join(BASE_DIR, "modelos_local", nome_pasta)
        print("salvando modelo localmente...")
        tokenizer.save_pretrained(caminho_salvar)
        model.save_pretrained(caminho_salvar)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho_json = os.path.join(BASE_DIR, 'lista_de_modelos.json')

    with open(caminho_json, 'r', encoding='utf-8') as arquivo:
        lista_modelos = json.load(arquivo)

    print("Escolha um modelo para baixar...")
    print("modelo:")
    for id, modelo in lista_modelos.items():
        print(f"id {id} do modelo: {modelo['Modelo']}")

    print("digite -1 para voltar")
    id_modelo = str(input("Digite o id do modelo: "))
    if id_modelo != "-1":
        baixando_modelo(lista_modelos[id_modelo]['Modelo'])
        with open(caminho_json, 'w', encoding='utf-8') as arquivo:
            json.dump(lista_modelos, arquivo, indent=4)
