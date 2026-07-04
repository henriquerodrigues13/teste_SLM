import json
import re
from Testando_modelos.schema_json import GeneratedQuestionsResponse


def extrair_e_validar(texto: str) -> tuple[bool, GeneratedQuestionsResponse | None, str | None]:
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
            ultimo_erro = f"JSON inválido: {e}"
            continue

        except Exception as e:
            ultimo_erro = f"Schema inválido: {e}"
            continue

    return False, None, ultimo_erro or "Nenhum candidato de JSON válido encontrado"


def parsear_json_bruto(texto: str) -> dict | str:
    match = re.search(r"```json\s*(.*?)\s*```", texto, re.DOTALL)
    json_str = match.group(1).strip() if match else texto.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return texto