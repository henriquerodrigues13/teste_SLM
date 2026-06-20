from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def baixando_modelo(modelo: str) -> str :
    try:
        print(f"baixando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(modelo)
        print(f"baixando modelo: {modelo}...")
        model = AutoModelForCausalLM.from_pretrained(modelo)
    except Exception as e:
        print(f"Erro: {e}")
    else:
        nome_pasta = modelo.replace("/", "--")
        print("salvando modelo localmente...")
        tokenizer.save_pretrained(f"./modelos_local/{nome_pasta}")
        model.save_pretrained(f"./modelos_local/{nome_pasta}")

if "__main__" == __name__:
    with open('lista_de_modelos.json', 'r', encoding='utf-8') as arquivo:
        lista_modelos = json.load(arquivo)

    print("Escolha um modelo para baixar...")
    print("modelo:")
    for id, modelo in lista_modelos.items():
        print(f"id {id} do modelo: {modelo['Modelo']}, modelo baixando: {"sim" if modelo['ModeloInstalado?'] else "não"}")


    id_modelo = str(input("Digite o id do modelo: "))
    baixando_modelo(lista_modelos[id_modelo]['Modelo'])
    lista_modelos[id_modelo]['ModeloInstalado?'] = True
    with open('lista_de_modelos.json', 'w', encoding='utf-8') as arquivo:
        json.dump(lista_modelos, arquivo, indent=4)
