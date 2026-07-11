import json
from pathlib import Path
from Testando_modelos.valida_json import extrair_e_validar, parsear_json_bruto
from Testando_modelos.instrucao.prompts import (
    construir_prompt_sistema,
    RESOLUCAO_NUMERADA,
    RESOLUCAO_OBJETIVA,
)


def cenario_d(modelo: str, tokenizer, model):
    import torch

    prompt_sistema = construir_prompt_sistema(RESOLUCAO_OBJETIVA)

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
            repetition_penalty=1.1,
        )

    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]

    print("output:")
    print("-" * 50)
    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("-" * 50)

    output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

    valido, resultado_validado, erro = extrair_e_validar(output_text)

    if valido:
        output_final = resultado_validado.model_dump()
    else:
        output_final = {
            "erro": erro,
            "output_bruto": parsear_json_bruto(output_text)
        }

    resultados = {
        "cenario_d": {
            "habilidades_testadas": ["EF08MA05", "EF06MA17"],
            "outputs": {
                "EF08MA05": output_final
            },
            "json_valido": {
                "EF08MA05": valido,
                "EF06MA17": None
            }
        }
    }

    prompt_sistema = construir_prompt_sistema(RESOLUCAO_OBJETIVA)

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
            repetition_penalty=1.1,
        )

    tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]

    print("output:")
    print("-" * 50)
    print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
    print("-" * 50)

    output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

    valido, resultado_validado, erro = extrair_e_validar(output_text)

    if valido:
        output_final = resultado_validado.model_dump()
    else:
        output_final = {
            "erro": erro,
            "output_bruto": parsear_json_bruto(output_text)
        }

    resultados["cenario_d"]["outputs"]["EF06MA17"] = output_final
    resultados["cenario_d"]["json_valido"]["EF06MA17"] = valido

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    nome_arquivo = modelo.replace("/", "--")
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    with open(arquivo, "r", encoding="utf-8") as f:
        dados_existentes = json.load(f)

    dados_existentes["cenario_d"] = resultados["cenario_d"]

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)


def cenario_d_gguf(llm, modelo):
    mensagens = [
        {"role": "system", "content": construir_prompt_sistema(RESOLUCAO_NUMERADA)},
        {"role": "user", "content": f"""
            Gere exatamente 1 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

            **Habilidade:** EF08MA05
            **Unidade Temática:** Geometria
        
            **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
            Não especificada.
        
            **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
            Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC.
        
            Gere 1 questão(ões) novas, inéditas e alinhadas à habilidade EF08MA05
            """}
    ]

    resposta = llm.create_chat_completion(
        messages=mensagens,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    output_text = resposta["choices"][0]["message"]["content"]

    print("output:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)

    valido, resultado, erro = extrair_e_validar(output_text)

    if valido:
        output_final = resultado.model_dump()
    else:
        output_final = {
            "erro": erro,
            "output_bruto": parsear_json_bruto(output_text)
        }

    resultados = {
        "cenario_d": {
            "habilidades_testadas": ["EF08MA05", "EF06MA17"],
            "outputs": {
                "EF08MA05": output_final
            },
            "json_valido": {
                "EF08MA05": valido,
                "EF06MA17": None
            }
        }
    }

    mensagens = [
        {"role": "system", "content": construir_prompt_sistema(RESOLUCAO_NUMERADA)},
        {"role": "user", "content": f"""
        Gere exatamente 1 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

        **Habilidade:** EF06MA17
        **Unidade Temática:** Geometria

        **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
        Não especificada.

        **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
        Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC.

        Gere 1 questão(ões) novas, inéditas e alinhadas à habilidade EF06MA17
        """}
    ]

    resposta = llm.create_chat_completion(
        messages=mensagens,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    output_text = resposta["choices"][0]["message"]["content"]

    print("output:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)

    valido, resultado, erro = extrair_e_validar(output_text)

    if valido:
        output_final = resultado.model_dump()
        resultados["cenario_d"]["json_valido"]["EF06MA17"] = True
    else:
        output_final = {
            "erro": erro,
            "output_bruto": parsear_json_bruto(output_text)
        }
        resultados["cenario_d"]["json_valido"]["EF06MA17"] = False

    resultados["cenario_d"]["outputs"]["EF06MA17"] = output_final

    pasta_resultados = Path(__file__).parent.parent / "resultados_gguf"
    nome_arquivo = modelo.replace("/", "--") + "-gguf"
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    if arquivo.exists() and arquivo.stat().st_size > 0:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_existentes = json.load(f)
    else:
        dados_existentes = {"modelo": modelo, "formato": "gguf", "eliminado": False, "motivo_eliminacao": None}

    dados_existentes["cenario_d"] = resultados["cenario_d"]

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)