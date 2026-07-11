import json
import re
from Testando_modelos.schema_json import GeneratedQuestionsResponse


def remover_think(texto: str) -> str:
    """Remove blocos de raciocínio <think>...</think> antes de procurar o JSON.

    Modelos de raciocínio (Qwen3, DeepSeek-R1, Phi-4-reasoning) emitem o
    pensamento em <think>...</think> antes da resposta. Removemos o bloco
    fechado; se ficou um <think> sem fechamento (resposta truncada), cortamos
    dali em diante — nesse caso não há JSON mesmo.
    """
    sem = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE)
    sem = re.sub(r"<think>.*$", "", sem, flags=re.DOTALL | re.IGNORECASE)
    return sem.strip()


def extrair_e_validar(texto: str) -> tuple[bool, GeneratedQuestionsResponse | None, str | None]:
    texto = remover_think(texto)
    candidatos = []

    matches_markdown = re.findall(r"```json\s*(.*?)\s*```", texto, re.DOTALL)
    candidatos.extend([m.strip() for m in matches_markdown])

    matches_chaves = re.findall(r"\{.*?\}(?=\s*(?:```|$|\{))", texto, re.DOTALL)
    candidatos.extend([m.strip() for m in matches_chaves])

    if not candidatos:
        return False, None, "JSON não encontrado na resposta"

    ultimo_erro = None

    for json_str in candidatos:
        try:
            dados = json.loads(json_str)
            resultado = GeneratedQuestionsResponse.model_validate(dados)
            return True, resultado, None

        except json.JSONDecodeError as e:
            ultimo_erro = "JSON inválido"
            continue

        except Exception as e:
            ultimo_erro = "Schema inválido"
            continue

    return False, None, ultimo_erro or "Nenhum candidato de JSON válido encontrado"


def parsear_json_bruto(texto: str) -> dict | str:
    texto = remover_think(texto)
    match = re.search(r"```json\s*(.*?)\s*```", texto, re.DOTALL)
    json_str = match.group(1).strip() if match else texto.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return texto


def validar_quantidade(resultado, quantidade_esperada: int) -> tuple[bool, str | None]:
    quantidade_real = len(resultado.questoes)

    if quantidade_real != quantidade_esperada:
        return False, f"Esperado {quantidade_esperada} questões, modelo gerou {quantidade_real}"

    return True, None
