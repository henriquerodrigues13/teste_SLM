import json
import re
from Testando_modelos.schema_json import GeneratedQuestionsResponse

def extrair_e_validar(texto: str) -> tuple[bool, GeneratedQuestionsResponse | None, str | None]:
    match = re.search(r"```json\s*(.*?)\s*```", texto, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
        else:
            return False, None, "JSON não encontrado na resposta"

    try:
        dados = json.loads(json_str)
        resultado = GeneratedQuestionsResponse.model_validate(dados)
        return True, resultado, None

    except json.JSONDecodeError as e:
        return False, None, f"JSON inválido: {e}"

    except Exception as e:
        return False, None, f"Schema inválido: {e}"