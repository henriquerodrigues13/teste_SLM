
# 🚀 Como Rodar Localmente

## Pré-requisitos

- Python 3.10 ou superior
- Git instalado
- Conta no [Hugging Face](https://huggingface.co)
- Mínimo 8 GB de RAM recomendado
- Mínimo 20 GB de espaço em disco

---

## 1. Clone o repositório

```bash
git clone https://github.com/henriquerodrigues13/teste_SML.git
cd teste_SML
```

---

## 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

---

## 3. Instale as dependências

```bash
pip install -e .
```

---

## 4. Autentique no Hugging Face

```bash
hf auth login
```

> Acesse [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), gere um token **read** e cole quando solicitado.

### Modelos que exigem acesso manual

Antes de baixar, acesse as páginas abaixo e aceite os termos de uso:

| Modelo | Link |
|---|---|
| Llama 3.2 | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Gemma 2 | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) |

---

## 5. Baixe os modelos

```bash
cd Testando_modelos
python download_modelos.py
```

O script vai exibir a lista de modelos disponíveis. Digite o ID do modelo que deseja baixar.

> - `HuggingFaceTB/SmolLM2-360M-Instruct`
> - `HuggingFaceTB/SmolLM2-1.7B-Instruct`
> - `Qwen/Qwen2.5-0.5B-Instruct`
> - `Qwen/Qwen2.5-1.5B-Instruct`
> - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## 6. Execute o benchmark

```bash
python .\Testando_modelos\main.py
```

O script vai:
1. Exibir a lista de modelos disponíveis
2. Pedir para você escolher o ID do modelo
3. Rodar o **Cenário A** (Cold Start)
4. Perguntar se deseja continuar com os demais cenários
5. Salvar os resultados em `resultados/<nome-do-modelo>.json`

---

## 7. Resultados

Os resultados ficam salvos em:

```
Testando_modelos/
└── resultados/
    ├── Qwen--Qwen2.5-0.5B-Instruct.json
    ├── HuggingFaceTB--SmolLM2-360M-Instruct.json
    └── ...
```

---

## ⚠️ Observações importantes

- Não feche o terminal durante a execução — isso interrompe o benchmark
- Mantenha o notebook em uma superfície ventilada durante os testes
- Modelos acima de 2B parâmetros podem ser lentos ou inviáveis em CPU puro
- Os modelos são salvos localmente — após baixar, os testes rodam **100% offline**
Os resultados ficam salvos em:
# 📁 Estrutura do Projeto — Testando_modelos


## Visão Geral

```
teste_SML/
└── Testando_modelos/
    ├── main.py
    ├── download_modelos.py
    ├── metricas.py
    ├── lista_de_modelos.json
    ├── cenarios/
    ├── instrucao/
    ├── modelos_local/
    └── resultados/
```

---

## 📄 Arquivos Raiz

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestrador principal — escolhe o modelo, chama os cenários em sequência e controla o fluxo de eliminação |
| `download_modelos.py` | Baixa e salva os modelos localmente em `modelos_local/` |
| `metricas.py` | Classe `MetricasInferencia` — mede RAM, tempo, tokens/segundo durante a inferência |
| `lista_de_modelos.json` | Lista de todos os modelos disponíveis para teste com seus IDs |

---

## 📁 cenarios/

Cada arquivo implementa um cenário de teste independente. Todos recebem `modelo`, `tokenizer`, `model` e `metricas` como parâmetros.

| Arquivo | Cenário | O que mede |
|---|---|---|
| `cenario_A.py` | Cold Start | Tempo de load + primeira inferência + RAM idle |
| `cenario_B.py` | Geração em Série | TPS, RAM, estabilidade em 10 rodadas consecutivas |
| `cenario_C.py` | *(a implementar)* | Taxa de JSON válido em português |
| `cenario_D.py` | *(a implementar)* | Qualidade sem contexto BNCC |
| `cenario_E.py` | *(a implementar)* | Raciocínio matemático passo a passo |

---

## 📁 instrucao/

Prompts específicos de cada cenário. Separados do código de execução para facilitar ajustes sem mexer na lógica.

| Arquivo | Uso |
|---|---|
| `instrucao_a.py` | Prompt do Cenário A |
| `instrucao_b.py` | Prompt do Cenário B |

---

## 📁 modelos_local/

Modelos salvos localmente após o download. Cada subpasta segue o padrão `autor--nome-do-modelo`.


modelos_local/
├── deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/
├── google--gemma-2-2b-it/
├── HuggingFaceTB--SmolLM2-360M-Instruct/
├── HuggingFaceTB--SmolLM2-1.7B-Instruct/
├── meta-llama--Llama-3.2-1B-Instruct/
├── Qwen--Qwen2.5-0.5B-Instruct/
├── Qwen--Qwen2.5-1.5B-Instruct/
├── Qwen--Qwen2.5-Math-1.5B-Instruct/
└── TinyLlama--TinyLlama-1.1B-Chat-v1.0/


Cada pasta contém:
- `model.safetensors` — pesos do modelo
- `config.json` — configurações da arquitetura
- `tokenizer.json` — vocabulário do tokenizer
- `tokenizer_config.json` — configurações do tokenizer
- `generation_config.json` — parâmetros padrão de geração
- `chat_template.jinja` — template de formatação do chat

---

## 📁 resultados/

Um JSON por modelo, gerado automaticamente após cada execução completa dos cenários.

```
resultados/
├── Qwen--Qwen2.5-0.5B-Instruct.json
├── Qwen--Qwen2.5-1.5B-Instruct.json
├── HuggingFaceTB--SmolLM2-360M-Instruct.json
└── ...
```

Cada JSON segue a estrutura:
```json
{
    "modelo": "...",
    "data": "...",
    "eliminado": false,
    "motivo_eliminacao": null,
    "eliminacao_manual": {
        "eliminado": false,
        "motivo": null,
        "eliminado_por": null
    },
    "cenario_a": { ... },
    "cenario_b": { ... },
    "cenario_c": { ... },
    "cenario_d": { ... },
    "cenario_e": { ... },
    "avaliacao_humana": {
        "status": "pendente",
        "aprovacao_pedagogica": { ... },
        "fidelidade_bncc": { ... }
    }
}
```

---

## 🔁 Fluxo de Execução

```
main.py
  │
  ├── usuário escolhe o modelo
  │
  ├── carrega modelo na memória
  │
  ├── cenario_A → salva JSON → pergunta se continua
  │
  └── se continua:
        ├── cenario_B
        ├── cenario_C
        ├── cenario_D
        └── cenario_E → salva JSON final
```

---

## ⚠️ Critérios de Eliminação Automática

| Critério | Limite | Cenário |
|---|---|---|
| RAM pico durante inferência | < 2 GB | A, B |
| Tempo de geração (3 questões) | < 90s | B |
| Crash / OOM em 20 execuções | 0 ocorrências | B |
| Taxa de aprovação pedagógica | ≥ 70% | D |
| Fidelidade BNCC | ≥ 90% | D |
| Resolução passo a passo correta | ≥ 80% | E |
| Funciona sem internet | Obrigatório | A |

| Arquivo | Uso |
|---|---|
| `instrucao_a.py` | Prompt do Cenário A |
| `instrucao_b.py` | Prompt do Cenário B |

---

## 📁 modelos_local/

Modelos salvos localmente após o download. Cada subpasta segue o padrão `autor--nome-do-modelo`.
