import json
from pathlib import Path
from Testando_modelos.instrucao.instrucao_b import seletor_de_questao
from Testando_modelos.valida_json import extrair_e_validar, validar_quantidade
from Testando_modelos.instrucao.prompts import (
    construir_prompt_sistema,
    RESOLUCAO_NUMERADA,
    RESOLUCAO_OBJETIVA,
)


def cenario_c(modelo: str, tokenizer, model):
    import torch

    execucoes = 0
    tentativas = 0
    json_valido = False
    while tentativas < 3 and not json_valido:
        execucoes += 1
        instrucao = seletor_de_questao()

        prompt_sistema = construir_prompt_sistema(RESOLUCAO_OBJETIVA)

        prompt = (
            f"""
            Gere exatamente 5 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

            **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
            **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

            **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
            {", ".join(instrucao["habilidadeContext"]["prerequisites"]) if instrucao["habilidadeContext"]["prerequisites"] else "Não especificada."}

            **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
            {"\n".join([f"Exemplo {x + 1}: {instrucao['habilidadeContext']['examples'][x]['enunciado']}"
                        for x in range(len(instrucao["habilidadeContext"]["examples"]))]) if instrucao["habilidadeContext"]["examples"] else "Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC."}

            Gere 5 questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
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
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        tokens_gerados = saida[0][inputs["input_ids"].shape[1]:]

        output_text = tokenizer.decode(tokens_gerados, skip_special_tokens=True)

        valido, resultado, erro = extrair_e_validar(output_text)

        if not valido:
            tentativas += 1
            continue
        else:
            json_valido = True

        print("output:")
        print("-" * 50)
        print(tokenizer.decode(tokens_gerados, skip_special_tokens=True))
        print("total de tentativas: ", tentativas)
        print("-" * 50)

    resultados = {
        "cenario_c": {
            "execucoes": execucoes ,
            "quantidade_de_questoes": 5,
            "tentativas": tentativas,
            "valido_primeira_tentativa": json_valido and tentativas == 0,
            "precisou_retry": json_valido and tentativas > 0,
            "falhou_com_3_tentativas": not json_valido,
            "aprovado": json_valido and tentativas <= 3,
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados"
    nome_arquivo = modelo.replace("/", "--")
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    with open(arquivo, "r", encoding="utf-8") as f:
        dados_existentes = json.load(f)

    dados_existentes["cenario_c"] = resultados["cenario_c"]

    if not json_valido:
        dados_existentes["eliminado"] = True
        dados_existentes["motivo_eliminacao"] = f"Falha de JSON recorrente — {tentativas} tentativas sem JSON válido (limite era 3)"

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)

def cenario_c_gguf(llm, modelo):
    execucoes = 0
    tentativas = 0
    json_valido = False
    while tentativas < 3 and not json_valido:
        execucoes += 1

        instrucao = seletor_de_questao()

        mensagens = [
            {"role": "system", "content": construir_prompt_sistema(RESOLUCAO_NUMERADA)},
            {"role": "user", "content": f"""
                    Gere exatamente 5 questão(ões) de múltipla escolha para a seguinte habilidade da BNCC:

                    **Habilidade:** {instrucao["habilidadeContext"]["habilidadeId"]}
                    **Unidade Temática:** {instrucao["habilidadeContext"]["unidadeTematica"]["nome"]}

                    **Pré-requisitos que o aluno já deve dominar (calibração de dificuldade):**
                    {", ".join(instrucao["habilidadeContext"]["prerequisites"]) if instrucao["habilidadeContext"]["prerequisites"] else "Não especificada."}

                    **Exemplos de questões existentes para esta habilidade (mantenha consistência de estilo):**
                    {"\n".join([f"Exemplo {x + 1}: {instrucao['habilidadeContext']['examples'][x]['enunciado']}"
                                for x in range(len(instrucao["habilidadeContext"]["examples"]))]) if instrucao["habilidadeContext"]["examples"] else "Nenhum exemplo disponível — crie questões seguindo as diretrizes gerais da BNCC."}

                    Gere 5 questão(ões) novas, inéditas e alinhadas à habilidade {instrucao["habilidadeContext"]["habilidadeId"]}.
                    """}
        ]

        resposta = llm.create_chat_completion(
            messages=mensagens,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
        )

        output_text = resposta["choices"][0]["message"]["content"]

        print("output:")
        print("-" * 50)
        print(output_text)
        print("total de tentativas: ", tentativas + 1)
        print("-" * 50)

        valido, resultado, erro = extrair_e_validar(output_text)

        if valido:
            quantidade_ok, erro_quantidade = validar_quantidade(resultado, 5)
            if not quantidade_ok:
                valido = False
                erro = erro_quantidade

        if not valido:
            tentativas += 1
            continue
        else:
            json_valido = True

        output_text = resposta["choices"][0]["message"]["content"]



    resultados = {
        "cenario_c": {
            "execucoes": execucoes,
            "quantidade_de_questoes": 5,
            "tentativas": tentativas,
            "valido_primeira_tentativa": json_valido and tentativas == 0,
            "precisou_retry": json_valido and tentativas > 0,
            "falhou_com_3_tentativas": not json_valido,
            "aprovado": json_valido and tentativas <= 3,
        }
    }

    pasta_resultados = Path(__file__).parent.parent / "resultados_gguf"
    nome_arquivo = modelo.replace("/", "--") + "-gguf"
    arquivo = pasta_resultados / f"{nome_arquivo}.json"

    if arquivo.exists() and arquivo.stat().st_size > 0:
        with open(arquivo, "r", encoding="utf-8") as f:
            dados_existentes = json.load(f)
    else:
        dados_existentes = {"modelo": modelo, "formato": "gguf", "eliminado": False, "motivo_eliminacao": None}

    dados_existentes["cenario_c"] = resultados["cenario_c"]

    if not json_valido:
        dados_existentes["eliminado"] = True
        dados_existentes["motivo_eliminacao"] = f"Falha de JSON recorrente — {tentativas} tentativas sem JSON válido (limite era 3)"

    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados_existentes, f, indent=4, ensure_ascii=False)