import datetime
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Testando_modelos.instrucao.instrucao_a import instrucao

def cenario_a(modelo: str, tokenizer, model, metricas):

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
        
        O campo "resolucao_passo_a_passo" é OBRIGATÓRIO. Liste apenas os passos de cálculo numerados, no formato "expressão = resultado". Sem texto introdutório, sem explicações teóricas, sem conclusão.
        """
    )

    prompt = (
        f"""
        Gere exatamente {instrucao["quantity"]} questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
        **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        {", ".join(instrucao["habilidadeContext"]["prerequisites"]) if instrucao["habilidadeContext"]["prerequisites"] else "Não especificada."}

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        {"Exemplo 1: " + instrucao["habilidadeContext"]["examples"][0]["enunciado"]}

        Gere {instrucao["quantity"]} questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
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

    metricas.iniciar(inputs["input_ids"].shape[1])
    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
        )
    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]
    metricas.finalizar(int(len(tokens_gerados)))

    print("output:")
    print("-" * 50)
    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("-"*50)
    print("relatorio:")
    print(metricas.relatorio())
    print("-" * 50)
    if metricas.ram_usada_mb > 2048.0:
        print("Modelo eliminado automaticamente")
        reposta = True
        motivo = f"RAM usada pelo modelo {metricas.ram_usada_mb:.1f} MB — limite era 2048 MB"
    else:
        print("valiação manual")
        reposta = input(f"O modelo {modelo} está eliminado? [S ou N]")
        if reposta.upper() == "S":
                motivo = "Português incompreensível"
                reposta = True
        else:
            reposta = False
            motivo = None
    descricao = input("Digite uma descrição ao modelo:")

    resultados = {
        "modelo": modelo,
        "data": datetime.datetime.now().strftime("%d/%m/%Y"),
        "eliminado": reposta,
        "motivo_eliminacao": motivo,
        "descrição": descricao ,
        "eliminacao_manual": {
            "eliminado": reposta,
            "eliminado_por": motivo,
        },
        "cenario_a": {
            "cold_start_segundos": f"{round(metricas.tempo_total, 2)} segundos",
            "ram_idle_mb": f"{round(metricas.ram_usada_mb, 1)} MB"
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    pasta_resultados.mkdir(exist_ok=True, parents=True)
    nome_arquivo = modelo.replace("/", "--")

    with open(pasta_resultados / f"{nome_arquivo}.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, indent=4, ensure_ascii=False)
