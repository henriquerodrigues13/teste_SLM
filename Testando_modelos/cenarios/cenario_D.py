import json
from pathlib import Path

import torch


def cenario_d(modelo: str, tokenizer, model):
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

        Retorne SOMENTE um JSON com esta estrutura exata:
        {
            "questoes": [
                {
                    "enunciado": "...",
                    "alternativas": {
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "...",
                        "E": "..."
                    },
                    "resposta_correta": "A",
                    "resolucao_passo_a_passo": "..."
                }
            ]
        }

         O campo "resolucao_passo_a_passo" é OBRIGATÓRIO. Mostre apenas os passos de cálculo para chegar na resposta correta, de forma objetiva.
        """
    )

    prompt = (
        f"""
        Gere exatamente 1 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** EF08MA05
        **Unidade Temática:** Geometria

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        Não especificada.

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC.

        Gere 1 questão(ões) novas, inéditas e alinhadas à habilidade EF08MA05
        """
    )

    modelos_sem_system = ["google/gemma-2-2b-it"]

    if modelo in modelos_sem_system:
        mensagens = [
            {"role": "user", "content": f"{prompt_sistema}\n\n{prompt}"}
        ]
    else:
        mensagens = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt},
        ]

    prompt = tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
        )

    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]

    print("output:")
    print("-" * 50)
    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("-" * 50)

    output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

    resultados = {
        "cenario_d": {
            "habilidades_testadas": ["EF08MA05", "EF06MA17"],
            "outputs": {
                "EF08MA05" : output_text
            },
            "json_valido": {
                "H3": None,
                "H5": None
            }
        }
    }

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

        Retorne SOMENTE um JSON com esta estrutura exata:
        {
            "questoes": [
                {
                    "enunciado": "...",
                    "alternativas": {
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "...",
                        "E": "..."
                    },
                    "resposta_correta": "A",
                    "resolucao_passo_a_passo": "..."
                }
            ]
        }

         O campo "resolucao_passo_a_passo" é OBRIGATÓRIO. Mostre apenas os passos de cálculo para chegar na resposta correta, de forma objetiva.
        """
    )

    prompt = (
        f"""
        Gere exatamente 1 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** EF06MA17
        **Unidade Temática:** Geometria

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        Não especificada.

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC.

        Gere 1 questão(ões) novas, inéditas e alinhadas à habilidade EF06MA17
        """
    )

    modelos_sem_system = ["google/gemma-2-2b-it"]

    if modelo in modelos_sem_system:
        mensagens = [
            {"role": "user", "content": f"{prompt_sistema}\n\n{prompt}"}
        ]
    else:
        mensagens = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt},
        ]

    prompt = tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
        )

    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]

    print("output:")
    print("-" * 50)
    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("-" * 50)

    output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

    resultados["cenario_d"]["outputs"]["EF06MA17"] = output_text

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    nome_arquivo = modelo.replace("/", "--")
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    with open(arquivo, "r", encoding="utf-8") as f:
        dados_existentes = json.load(f)

    dados_existentes["cenario_d"] = resultados["cenario_d"]

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)
