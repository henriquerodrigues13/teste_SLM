import json
import os
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def gerador_de_questoes(modelo: str, instrucao):
    print("Carregando modelo...")
    tempo_inicial = time.time()

    BASE_DIR = Path(__file__).parent
    nome_pasta = modelo.replace("/", "--")
    caminho = BASE_DIR / "modelos_local" / nome_pasta

    tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(caminho, local_files_only=True)
    model.eval()
    tempo_final = time.time() - tempo_inicial
    print(f"Pronto! demorou : {tempo_final:.2f} segundos")

    prompt_sistema = (
        """
        Você é um especialista sênior em elaboração de itens avaliativos de Matemática para o Ensino Fundamental (1º ao 9º ano), com domínio profundo da Base Nacional Comum Curricular (BNCC).

        Suas responsabilidades:
        - Criar questões de múltipla escolha (5 alternativas: A, B, C, D, E) rigorosamente alinhadas à habilidade da BNCC solicitada.
        - Garantir que o enunciado seja claro, contextualizado e adequado à faixa etária do estudante.
        - Produzir alternativas plausíveis e com distratores pedagogicamente fundamentados (erros comuns dos alunos, não respostas absurdas).
        - Redigir a resolução passo a passo de forma didática, como faria um professor explicando para o aluno.
        - Calibrar a dificuldade respeitando os pré-requisitos informados: os conceitos listados em pré-requisitos devem ser dominados pelo aluno, e NÃO devem ser o foco central da questão — use-os como base para atingir a habilidade-alvo.
        - Manter consistência de estilo, formato e nível de abstração com os exemplos fornecidos (few-shot).
        
        Regras absolutas:
        - Retorne SOMENTE o JSON estruturado solicitado, sem texto adicional, markdown ou explicações fora do JSON.
        - Todas as questões devem ser inéditas entre si na mesma resposta.
        - A resposta correta deve ser distribuída de forma variada entre A, B, C, D e E ao longo das questões.
        """
    )

    prompt = (
        f"""
        Gere exatamente {instrucao["quantity"]} questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
        **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        {", ".join(instrucao["habilidadeContext"]["prerequisites"])}

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        {"Exemplo 1: " + instrucao["habilidadeContext"]["examples"][0]["enunciado"]}

        Gere {instrucao["quantity"]} questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
        """
    )

    mensagens = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    tempo_inicial = time.time()
    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,
        )
    tempo_final = time.time() - tempo_inicial
    print(f"demorou : {tempo_final:.2f} segundos")
    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(tokens_gerados, skip_special_tokens=True)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho_json = os.path.join(BASE_DIR, 'lista_de_modelos.json')
    instrucao = dict(classId="00000000-0000-0000-0000-000000000000", quantity=1,
         habilidadeContext={"habilidadeId": "EF09MA01", "unidadeTematica": {"nome": "Números"},
                            "prerequisites": ["EF08MA01", "EF07MA01"], "examples": [{"idPostgres": "q-exemplo-001",
                                                                                     "enunciado": "Dentre os números abaixo, qual NÃO pode ser escrito na forma de uma fração a/b (com a e b inteiros, b≠0)?",
                                                                                     "alternativas": {"A": "0,5",
                                                                                                      "B": "3/4", "C": "√2",
                                                                                                      "D": "-2",
                                                                                                      "E": "0,333..."},
                                                                                     "gabarito": "C"}]}, _golden={
            "questoes": [
                {"enunciado": "Considere os números: 3,14; 1/2; √2; 0,333... e -7. Qual deles é um número irracional?",
                 "alternativas": {"A": "3,14", "B": "1/2", "C": "√2", "D": "0,333...", "E": "-7"}, "resposta_correta": "C",
                 "resolucao_passo_a_passo": "Um número irracional não pode ser escrito como fração a/b (a e b inteiros, b≠0). Analisando: 3,14 = 314/100 é racional; 1/2 é racional; √2 não pode ser expresso como fração de inteiros (sua representação decimal é infinita e não periódica) — portanto é irracional; 0,333... = 1/3 é racional; -7 = -7/1 é racional. Logo, a resposta é √2."}]})

    with open(caminho_json, 'r', encoding='utf-8') as arquivo:
        lista_modelos = json.load(arquivo)

    id_modelo = str(input("Digite o id do modelo: "))
    print(f"{gerador_de_questoes(lista_modelos[id_modelo]['Modelo'],instrucao)}")
    x = input("digite para volta ao inicio")