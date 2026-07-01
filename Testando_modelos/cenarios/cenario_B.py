import json
from pathlib import Path
import torch
from Testando_modelos.valida_json import extrair_e_validar
from Testando_modelos.instrucao.instrucao_b import seletor_de_questao


def cenario_b(modelo: str, tokenizer, model, metricas):
    crashes_oom = 0
    json_validos = 0
    json_invalidos = 0
    for i in range(10):
        metricas.iniciar_rodada()
        tokens_gerados_por_rodada = 0

        instrucao = seletor_de_questao()

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
            Gere exatamente 3 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

            **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
            **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

            **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
            {", ".join(instrucao["habilidadeContext"]["prerequisites"]) if instrucao["habilidadeContext"]["prerequisites"] else "Não especificada."}

            **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
            {"\n".join([f"Exemplo {x + 1}: {instrucao['habilidadeContext']['examples'][x]['enunciado']}"
            for x in range(len(instrucao["habilidadeContext"]["examples"]))]) if instrucao["habilidadeContext"]["examples"] else "Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC."}

            Gere 3 questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
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

        try:
            with torch.no_grad():
                saida = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.3,
                )
        except MemoryError:
            crashes_oom += 1

        except Exception as e:
            crashes_oom += 1
            print(f"Crash na rodada {i}: {e}")

        tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]
        tokens_gerados_por_rodada += len(tokens_gerados)

        output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

        print("output:")
        print("-" * 50)
        print(output_text)
        print("-" * 50)

        valido, resultado, erro = extrair_e_validar(output_text)

        if valido:
            json_validos += 1
        else:
            json_invalidos += 1
        metricas.finalizar_rodada(tokens_gerados_por_rodada)
    dados = metricas.media_rodadas()
    print("-" * 50)
    print("Relatório :")
    print(f"""  
      Total de rodadas     : 10
      JSON válidos         : {json_validos}
      JSON inválidos       : {json_invalidos}
      Crashes/OOM          : {crashes_oom}
      TPS médio            : {dados['tps_medio']:.2f} t/s
      RAM média            : {dados['ram_media_mb']:.1f} MB
      Tempo médio/rodada   : {dados['tempo_medio_segundos']:.2f}s""")
    print("-" * 50)

    resultados = {
        "cenario_b": {
            "rodadas": 10,
            "json_validos": json_validos,
            "json_invalidos": json_invalidos,
            "taxa_json_valido_percent": round(json_validos / 100, 1),
            "crashes_oom": crashes_oom,
            "tps_medio": round(dados["tps_medio"], 2),
            "ram_media_mb": round(dados["ram_media_mb"], 1),
            "tempo_medio_por_rodada_segundos": round(dados["tempo_medio_segundos"], 2),
            "aprovado_crashes": crashes_oom == 0,
            "aprovado_ram": dados["ram_media_mb"] < 2048,
            "aprovado_tempo": dados["tempo_medio_segundos"] < 90,
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    nome_arquivo = modelo.replace("/", "--")
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    with open(arquivo, "r", encoding="utf-8") as f:
        dados_existentes = json.load(f)

    dados_existentes["cenario_b"] = resultados["cenario_b"]

    if crashes_oom > 0:
        dados_existentes["eliminado"] = True
        dados_existentes["motivo_eliminacao"] = f"Crash/OOM detectado — {crashes_oom} ocorrência(s)"

    elif max(metricas.TPS_por_rodada) > 2048:
        dados_existentes["eliminado"] = True
        dados_existentes["motivo_eliminacao"] = f"RAM média {dados['ram_media_mb']:.1f} MB — limite era 2048 MB"

    elif len(metricas.TPS_por_rodada) == 10:
        tps_inicial = metricas.TPS_por_rodada[0]
        tps_final = metricas.TPS_por_rodada[-1]
        if tps_inicial > 0:
            degradacao = (tps_inicial - tps_final) / tps_inicial * 100
            if degradacao > 40:
                dados_existentes["eliminado"] = True
                dados_existentes["motivo_eliminacao"] = f"Degradação de TPS severa: {degradacao:.1f}% entre 1ª e 10ª geração"

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)
