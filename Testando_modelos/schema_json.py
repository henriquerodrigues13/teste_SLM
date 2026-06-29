from pydantic import BaseModel, Field
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

class GeneratedQuestionsResponse(BaseModel):
    questoes: list[Questao] = Field(min_length=1)