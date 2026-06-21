import os
from Testando_modelos import download_modelos, config_teste

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
        os.system("cls")
        config_teste.main()
