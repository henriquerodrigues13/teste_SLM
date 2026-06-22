import json
import os
from time import sleep

from Testando_modelos.cenarios import *
from Testando_modelos import download_modelos, config_teste
from Testando_modelos.cenarios.cenario_A import cenario_a

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
        cenario_a(lista_modelos[x]['Modelo'])

