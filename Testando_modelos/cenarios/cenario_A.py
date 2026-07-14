import datetime
import json
from pathlib import Path
from Testando_modelos.instrucao.instrucao_a import instrucao
from Testando_modelos.instrucao.prompts import (
    construir_prompt_sistema,
    RESOLUCAO_NUMERADA,
    RESOLUCAO_CORRIDA,
    sufixo_no_think,
)


def cenario_a(modelo: str, tokenizer, model, metricas):
    import torch


    prompt_sistema = construir_prompt_sistema(RESOLUCAO_NUMERADA)

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
    if metricas.ram_pico_mb > 2048.0:
        print("Modelo eliminado automaticamente")
        reposta = True
        motivo = f"RAM pico do processo {metricas.ram_pico_mb:.1f} MB — limite era 2048 MB"
    else:
        print("avaliação manual")
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
            "ram_idle_mb": f"{round(metricas.ram_inicio_mb, 1)} MB",
            "ram_pico_mb": f"{round(metricas.ram_pico_mb, 1)} MB"
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    pasta_resultados.mkdir(exist_ok=True, parents=True)
    nome_arquivo = modelo.replace("/", "--")

    with open(pasta_resultados / f"{nome_arquivo}.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, indent=4, ensure_ascii=False)


def cenario_a_gguf(llm, modelo, metricas):

    prompt_sistema = construir_prompt_sistema(RESOLUCAO_CORRIDA)

    prompt = f"""
        Gere exatamente {instrucao["quantity"]} questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
        **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        {", ".join(instrucao["habilidadeContext"]["prerequisites"]) if instrucao["habilidadeContext"]["prerequisites"] else "Não especificada."}

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        {"Exemplo 1: " + instrucao["habilidadeContext"]["examples"][0]["enunciado"]}

        Gere {instrucao["quantity"]} questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
        """

    modelos_sem_system = ["gemma-2-2b-it"]
    prompt = prompt + sufixo_no_think(modelo)

    if any(nome in modelo for nome in modelos_sem_system):
        mensagens = [
            {"role": "user", "content": f"{prompt_sistema}\n\n{prompt}"}
        ]
    else:
        mensagens = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt},
        ]

    metricas.marcar_inicio_resposta()
    resposta = llm.create_chat_completion(
        messages=mensagens,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        repeat_penalty=1.1,
    )

    tokens_saida = resposta["usage"]["completion_tokens"]
    tokens_entrada = resposta["usage"]["prompt_tokens"]

    metricas.tokens_entrada = tokens_entrada
    metricas.finalizar(tokens_saida)

    output_text = resposta["choices"][0]["message"]["content"]

    print("output:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)
    print(metricas.relatorio())
    print("-" * 50)

    resultados = {
        "cenario_a": {
            "cold_start_segundos": round(metricas.tempo_total, 2),
            "ram_idle_mb": round(metricas.ram_inicio_mb, 1),
            "ram_pico_mb": round(metricas.ram_pico_mb, 1),
            "ram_usada_mb": round(metricas.ram_usada_mb, 1),
            "tokens_por_segundo": round(metricas.tokens_por_segundo, 2),
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados_gguf"
    pasta_resultados.mkdir(exist_ok=True)
    nome_arquivo = modelo.replace("/", "--") + "-gguf"
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    dados_existentes = {"modelo": modelo, "formato": "gguf", "eliminado": False, "motivo_eliminacao": None}
    if arquivo.exists():
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_existentes = json.load(f)

    dados_existentes["cenario_a"] = resultados["cenario_a"]

    if metricas.ram_pico_mb > 2048:
        dados_existentes["eliminado"] = True
        dados_existentes["motivo_eliminacao"] = f"RAM pico do processo {metricas.ram_pico_mb:.1f} MB — limite era 2048 MB"
        dados_existentes["eliminacao_manual"] = {"eliminado": True, "eliminado_por": None}
    else:
        print("avaliação manual")
        resposta = input(f"O modelo {modelo} está eliminado? [S ou N]: ")

        if resposta.upper() == "S":
            dados_existentes["eliminado"] = True
            dados_existentes["motivo_eliminacao"] = "Português incompreensível"
            dados_existentes["eliminacao_manual"] = {"eliminado": True, "eliminado_por": "Português incompreensível"}
        else:
            dados_existentes["eliminacao_manual"] = {"eliminado": False, "eliminado_por": None}
    descricao = input("Descrição adicional (opcional, Enter pra pular): ")
    dados_existentes["descrição"] = descricao if descricao else None

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)
