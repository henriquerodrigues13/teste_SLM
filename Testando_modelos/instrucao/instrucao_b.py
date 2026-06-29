import random


def seletor_de_questao():
    instrucoes = [
        {"_meta": {"tipo": "completo", "few_shot": 1, "golden": True,
                   "nota": "Questões few-shot e golden são SINTÉTICAS/ILUSTRATIVAS. Para golden real, usar questões APPROVED da tabela questions (Postgres)."},
         "classId": "00000000-0000-0000-0000-000000000000", "quantity": 1,
         "habilidadeContext": {"habilidadeId": "EF09MA01", "unidadeTematica": {"nome": "Números"},
                               "prerequisites": ["EF08MA01", "EF07MA01"], "examples": [{"idPostgres": "q-exemplo-001",
                                                                                        "enunciado": "Dentre os números abaixo, qual NÃO pode ser escrito na forma de uma fração a/b (com a e b inteiros, b≠0)?",
                                                                                        "alternativas": {"A": "0,5",
                                                                                                         "B": "3/4",
                                                                                                         "C": "√2",
                                                                                                         "D": "-2",
                                                                                                         "E": "0,333..."},
                                                                                        "gabarito": "C"}]}, "_golden": {
            "questoes": [
                {"enunciado": "Considere os números: 3,14; 1/2; √2; 0,333... e -7. Qual deles é um número irracional?",
                 "alternativas": {"A": "3,14", "B": "1/2", "C": "√2", "D": "0,333...", "E": "-7"},
                 "resposta_correta": "C",
                 "resolucao_passo_a_passo": "Um número irracional não pode ser escrito como fração a/b (a e b inteiros, b≠0). Analisando: 3,14 = 314/100 é racional; 1/2 é racional; √2 não pode ser expresso como fração de inteiros (sua representação decimal é infinita e não periódica) — portanto é irracional; 0,333... = 1/3 é racional; -7 = -7/1 é racional. Logo, a resposta é √2."}]}},
        {"_meta": {"tipo": "completo", "few_shot": 2, "golden": False},
         "classId": "00000000-0000-0000-0000-000000000000", "quantity": 3,
         "habilidadeContext": {"habilidadeId": "EF06MA14", "unidadeTematica": {"nome": "Álgebra"},
                               "prerequisites": ["EF06MA01"], "examples": [
                 {"idPostgres": "q-exemplo-002", "enunciado": "Se x + 5 = 12, qual é o valor de x?"},
                 {"idPostgres": "q-exemplo-003",
                  "enunciado": "Se 2x = 10, determine o valor de x aplicando as propriedades da igualdade."}]},
         "_golden": None},
        {"_meta": {"tipo": "completo", "few_shot": 1, "golden": False},
         "classId": "00000000-0000-0000-0000-000000000000", "quantity": 2,
         "habilidadeContext": {"habilidadeId": "EF02MA01", "unidadeTematica": {"nome": "Números"}, "prerequisites": [],
                               "examples": [{"idPostgres": "q-exemplo-004",
                                             "enunciado": "Usando a reta numérica, qual número está entre 23 e 27?"}]},
         "_golden": None}]
    x = random.randint(0,2)
    return instrucoes[x]
