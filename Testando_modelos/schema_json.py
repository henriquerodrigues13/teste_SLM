from pydantic import BaseModel, Field, field_validator
from typing import Literal

class Alternativas(BaseModel):
    A: str
    B: str
    C: str
    D: str
    E: str

class Questao(BaseModel):
    enunciado: str = Field(min_length=1)
    alternativas: Alternativas
    resposta_correta: Literal["A", "B", "C", "D", "E"]
    resolucao_passo_a_passo: str = Field(min_length=1)

    @field_validator("resolucao_passo_a_passo", mode="before")
    @classmethod
    def juntar_lista_em_string(cls, valor):
        if isinstance(valor, list):
            itens_validos = [str(item) for item in valor if item is not None and str(item).strip()]
            return "\n".join(itens_validos)
        return valor

class GeneratedQuestionsResponse(BaseModel):
    questoes: list[Questao] = Field(min_length=1)