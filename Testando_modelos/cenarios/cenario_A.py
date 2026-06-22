from pathlib import Path
from time import sleep

import torch
from Testando_modelos.metricas import MetricasInferencia
from transformers import AutoTokenizer, AutoModelForCausalLM
from Testando_modelos.instrucao.instrucao_a import instrucao

def cenario_a(modelo: str):
    print("Carregando modelo...")
    metricas = MetricasInferencia()
    BASE_DIR = Path(__file__).parent.parent
    nome_pasta = modelo.replace("/", "--")
    caminho = BASE_DIR / "modelos_local" / nome_pasta

    metricas.tempo_load_modelo_inicial
    tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(caminho, local_files_only=True)
    model.eval()
    metricas.tempo_load_modelo_final
    print(f"Pronto! modelo: {modelo} carregando")

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

    metricas.iniciar(inputs["input_ids"].shape[1])
    with torch.no_grad():
        saida = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,
        )
    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]
    metricas.finalizar(int(len(tokens_gerados)))  # ← depois de calcular tokens_gerados

    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("")
    print(metricas.relatorio())
    sleep(30)
