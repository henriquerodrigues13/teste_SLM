# 🚀 Como Rodar Localmente

## Pré-requisitos

- Python 3.10 ou superior
- Git instalado
- Conta no [Hugging Face](https://huggingface.co)
- Mínimo 8 GB de RAM recomendado
- Mínimo 20 GB de espaço em disco
- **Windows:** Visual Studio Build Tools (necessário para compilar `llama-cpp-python`) — veja seção abaixo

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

> ⚠️ **Windows:** esse comando pode falhar ao instalar `llama-cpp-python` com erro de `nmake`/`CMAKE_C_COMPILER not set`. Se isso acontecer, siga a seção **"Compilando llama-cpp-python no Windows"** abaixo antes de continuar.

---

## 4. Compilando `llama-cpp-python` no Windows

O `llama-cpp-python` precisa compilar código C++ na máquina. Os wheels pré-compilados disponíveis publicamente nem sempre são compatíveis com todo processador (podem assumir suporte a AVX-512, que CPUs mais recentes de notebook — como as linhas Intel Core U — não têm, causando o erro `OSError: [WinError -1073741795] 0xc000001d`).

**4.1. Instale o Visual Studio Build Tools**

Baixe em: https://visualstudio.microsoft.com/visual-cpp-build-tools/

No instalador, marque o workload **"Desktop development with C++"**.

**4.2. Abra o "Developer Command Prompt for VS"**

Procure por esse terminal no menu Iniciar (ele já vem com o compilador configurado no PATH).

**4.3. Ative o venv e compile**

```cmd
cd caminho\para\teste_SML
.venv\Scripts\activate.bat

set CMAKE_ARGS=-DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --no-cache-dir --force-reinstall --no-binary llama-cpp-python
```

> As flags acima ativam AVX2 (suportado pela maioria dos CPUs modernos) e desativam AVX-512, evitando o erro de instrução ilegal. Ajuste conforme o seu processador se necessário.

A compilação leva alguns minutos. Depois disso, o restante das dependências do `pyproject.toml` pode ser instalado normalmente com `pip install -e .`.

---

## 5. Autentique no Hugging Face

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

## 6. Baixe os modelos

O projeto tem dois formatos de modelo: **base** (safetensors, via `transformers`) e **GGUF** (via `llama-cpp-python`). O foco atual do projeto é GGUF, mas o fluxo de modelos base ainda é suportado.

### Modelos GGUF (recomendado)

```bash
cd Testando_modelos
python download_modelos_gguf.py
```

O script lê `lista_de_modelos_gguf.json`, exibe os modelos disponíveis com sua quantização, e baixa o arquivo `.gguf` escolhido para `modelos_local_gguf/<autor>--<nome-do-modelo>/`.

> Modelos disponíveis incluem variações de Qwen2.5 (0.5B a 7B), SmolLM2/SmolLM3, DeepSeek-R1-Distill-Qwen-1.5B, TinyLlama, Llama 3.2 (1B/3B), Gemma 2/3, Phi-3/3.5, Granite 3.0, Mistral 7B, Llama 3.1 8B e WizardMath 7B — todos em quantização Q4_K_M (ou equivalente).

### Modelos base (opcional)

```bash
python download_modelos.py
```

Baixa os pesos completos (safetensors) para `modelos_local/<autor>--<nome-do-modelo>/`.

---

## 7. Execute o benchmark

```bash
python .\Testando_modelos\main.py
```

O script exibe um menu com 5 opções:

| Opção | Ação |
|---|---|
| `0` | Sair do sistema |
| `1` | Baixar modelos base |
| `2` | Baixar modelos GGUF |
| `3` | Testar modelos base |
| `4` | Testar modelos GGUF |

Ao escolher testar um modelo (opção `3` ou `4`), o fluxo é:
1. Exibir a lista de modelos já baixados
2. Pedir para você escolher o ID do modelo
3. Carregar o modelo na memória
4. Rodar o **Cenário A** (Cold Start)
5. Perguntar se deseja continuar com os demais cenários (B, C, D)
6. Salvar os resultados em `resultados/<nome-do-modelo>.json`

> O **Cenário E** (avaliação humana — aprovação pedagógica e fidelidade BNCC) não é executado pelo `main.py`. É um processo de revisão manual, feito posteriormente sobre os resultados salvos.

---

## 8. Resultados

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

---

# 📁 Estrutura do Projeto — Testando_modelos

## Visão Geral

```
teste_SML/
└── Testando_modelos/
    ├── main.py
    ├── download_modelos.py
    ├── download_modelos_gguf.py
    ├── metricas.py
    ├── lista_de_modelos.json
    ├── lista_de_modelos_gguf.json
    ├── cenarios/
    ├── instrucao/
    ├── modelos_local/          (opcional — modelos base)
    ├── modelos_local_gguf/     (modelos GGUF)
    └── resultados/
```

---

## 📄 Arquivos Raiz

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestrador principal — menu de 5 opções (baixar/testar modelos base ou GGUF), controla o fluxo de cenários e eliminação |
| `download_modelos.py` | Baixa e salva modelos base (safetensors) localmente em `modelos_local/` |
| `download_modelos_gguf.py` | Baixa e salva modelos GGUF localmente em `modelos_local_gguf/`, a partir de `lista_de_modelos_gguf.json` |
| `metricas.py` | Classe `MetricasInferencia` — mede RAM, tempo, tokens/segundo durante a inferência |
| `lista_de_modelos.json` | Lista de modelos base disponíveis para teste com seus IDs |
| `lista_de_modelos_gguf.json` | Lista de modelos GGUF disponíveis, com repositório, arquivo `.gguf` e quantização |

---

## 📁 cenarios/

Cada arquivo implementa um cenário de teste independente. Todos recebem `modelo`, `tokenizer`/`llm` e `metricas` como parâmetros.

| Arquivo | Cenário | O que mede | Status |
|---|---|---|---|
| `cenario_A.py` | Cold Start | Tempo de load + primeira inferência + RAM idle | ✅ Implementado |
| `cenario_B.py` | Geração em Série | TPS, RAM, estabilidade em 10 rodadas consecutivas | ✅ Implementado |
| `cenario_C.py` | Formato JSON | Taxa de JSON válido em português | ✅ Implementado |
| `cenario_D.py` | Qualidade sem contexto BNCC | Qualidade das respostas sem contexto curricular | ✅ Implementado |
| `cenario_E.py` | Raciocínio matemático passo a passo | Validação humana (aprovação pedagógica e fidelidade BNCC) | 🕒 Manual — não automatizado |

---

## 📁 instrucao/

Prompts específicos de cada cenário. Separados do código de execução para facilitar ajustes sem mexer na lógica.

| Arquivo | Uso |
|---|---|
| `instrucao_a.py` | Prompt do Cenário A |
| `instrucao_b.py` | Prompt do Cenário B |

---

## 📁 modelos_local_gguf/

Modelos GGUF salvos localmente após o download via `download_modelos_gguf.py`. Cada subpasta segue o padrão `autor--nome-do-modelo`, contendo o arquivo `.gguf` correspondente.

```
modelos_local_gguf/
├── Qwen--Qwen2.5-0.5B-Instruct-GGUF/
├── Qwen--Qwen2.5-1.5B-Instruct-GGUF/
├── Qwen--Qwen2.5-3B-Instruct-GGUF/
├── HuggingFaceTB--SmolLM2-1.7B-Instruct-GGUF/
├── bartowski--DeepSeek-R1-Distill-Qwen-1.5B-GGUF/
├── bartowski--Llama-3.2-1B-Instruct-GGUF/
└── ...
```

## 📁 modelos_local/ (opcional)

Usado apenas quando modelos base (safetensors) são baixados via `download_modelos.py`. Cada pasta contém `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` e `chat_template.jinja`.

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
  ├── [0] Sair
  ├── [1] Baixar modelos base
  ├── [2] Baixar modelos GGUF
  ├── [3] Testar modelos base ──┐
  └── [4] Testar modelos GGUF ──┤
                                 │
                                 ├── usuário escolhe o modelo
                                 ├── carrega modelo na memória
                                 ├── cenario_A → salva JSON → pergunta se continua
                                 └── se continua:
                                       ├── cenario_B
                                       ├── cenario_C
                                       └── cenario_D → salva JSON

  (cenario_E roda separadamente, como revisão humana manual)
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
| Resolução passo a passo correta | ≥ 80% | E (avaliação manual) |
| Funciona sem internet | Obrigatório | A |
