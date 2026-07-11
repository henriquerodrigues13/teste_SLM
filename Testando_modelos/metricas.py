import time
import os
import psutil
#import torch
import threading

class MetricasInferencia:
    def __init__(self):
        self.processo = psutil.Process(os.getpid())
        self._monitorando = False
        self._thread = None
        self.tempo_load_inicial = 0
        self.tempo_load_final = 0
        self.tempo_inicio = None
        self.tempo_fim = None
        self.ram_pico_mb = 0
        self.ram_inicio_mb = 0
        self.tokens_entrada = 0
        self.tokens_saida = 0
        self.tempo_total_por_rodada = []
        self.TPS_por_rodada = []
        self.ram_por_rodada = []

    def _monitorar_ram(self):
        while self._monitorando:
            ram_atual = self.processo.memory_info().rss / (1024 ** 2)
            if ram_atual > self.ram_pico_mb:
                self.ram_pico_mb = ram_atual
            time.sleep(0.1)

    def iniciar(self, tokens_entrada: int):
        self.tokens_entrada = tokens_entrada
        self.ram_inicio_mb = self.processo.memory_info().rss / (1024 ** 2)
        self.ram_pico_mb = self.ram_inicio_mb
        self.tempo_inicio = time.perf_counter()
        self._monitorando = True
        self._thread = threading.Thread(target=self._monitorar_ram, daemon=True)
        self._thread.start()

    def finalizar(self, tokens_saida: int):
        self.tempo_fim = time.perf_counter()
        self.tokens_saida = tokens_saida
        self._monitorando = False
        self._thread.join()

    def iniciar_rodada(self):
        self.tempo_inicio = time.perf_counter()
        self.ram_inicio_mb = self.processo.memory_info().rss / (1024 ** 2)
        self.ram_pico_mb = self.ram_inicio_mb
        self._monitorando = True
        self._thread = threading.Thread(target=self._monitorar_ram, daemon=True)
        self._thread.start()

    def finalizar_rodada(self, tokens_saida: int):
        self.tempo_fim = time.perf_counter()
        self.tokens_saida = tokens_saida
        self._monitorando = False
        self._thread.join()
        # Guarda o pico ABSOLUTO de RAM da rodada (inclui os pesos do modelo já
        # carregados), não o delta — o orçamento de < 2 GB é sobre o RSS total.
        self.ram_por_rodada.append(self.ram_pico_mb)
        self.TPS_por_rodada.append(tokens_saida / self.tempo_resposta_modelo)
        self.tempo_total_por_rodada.append(self.tempo_fim - self.tempo_inicio)

    def tempo_load_modelo_inicial(self):
        self.tempo_load_inicial = time.perf_counter()

    def tempo_load_modelo_final(self):
        self.tempo_load_final = time.perf_counter()

    @property
    def tempo_carregamento_modelo(self) -> float:
        return self.tempo_load_final - self.tempo_load_inicial

    @property
    def tempo_resposta_modelo(self) -> float:
        return self.tempo_fim - self.tempo_inicio

    @property
    def tempo_total(self) -> float:
        return self.tempo_resposta_modelo + self.tempo_carregamento_modelo

    @property
    def tokens_por_segundo(self) -> float:
        return self.tokens_saida / self.tempo_total

    @property
    def ram_usada_mb(self) -> float:
        return self.ram_pico_mb - self.ram_inicio_mb

    @property
    def media_rodadas(self):
        tps_medio = sum(self.TPS_por_rodada) / len(self.TPS_por_rodada)
        ram_media = sum(self.ram_por_rodada) / len(self.ram_por_rodada)
        ram_pico = max(self.ram_por_rodada) if self.ram_por_rodada else 0
        tempo_medio = sum(self.tempo_total_por_rodada) / len(self.tempo_total_por_rodada)

        return {
            "tps_medio": tps_medio,
            "ram_media_mb": ram_media,
            "ram_pico_mb": ram_pico,
            "tempo_medio_segundos": tempo_medio
        }


    def relatorio(self) -> str:
        return (
            f"""
            Tempo de carregamento do modelos        : {self.tempo_carregamento_modelo:.2f}s
            Tempo de resposta do modelo             : {self.tempo_resposta_modelo:.2f}s
            Tempo total                             : {self.tempo_total:.2f}s
            Tokens entrada                          : {self.tokens_entrada}
            Tokens gerados                          : {self.tokens_saida}
            Tokens/segundo                          : {self.tokens_por_segundo:.2f}
            RAM início                              : {self.ram_inicio_mb:.1f} MB
            RAM pico                                : {self.ram_pico_mb:.1f} MB
            RAM usada (delta)                       : {self.ram_usada_mb:.1f} MB
            """
        )
