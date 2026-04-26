import numpy as np
from laboratorio3_kedro.pipelines.histogram.analyzer import ImageAnalyzer

def test_histograma_tiene_256_posiciones():
    """Verifica que el histograma tenga 256 posiciones."""
    analyzer = ImageAnalyzer()
    imagen = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    histograma = analyzer.calcular_histograma(imagen)
    assert len(histograma) == 256

def test_clasificacion_retorna_valor_logico():
    """Valida que el método de análisis retorne un resultado lógico."""
    analyzer = ImageAnalyzer()
    imagen = np.zeros((100, 100), dtype=np.uint8)
    histograma = analyzer.calcular_histograma(imagen)
    clasificacion = analyzer.clasificar_imagen(histograma)
    assert clasificacion in ["oscura", "clara"]