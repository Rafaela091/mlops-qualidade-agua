from mlops_demo.config import ALVO, CONTRATO_LIMIARES, FEATURES


def test_config_variaveis_existem():
    assert isinstance(FEATURES, list)
    assert isinstance(ALVO, str)
    assert isinstance(CONTRATO_LIMIARES, dict)


def test_config_features_agua_especifico():
    # Teste NOVO: Garante que o projeto está configurado exatamente para os 8 atributos químicos da água
    assert len(FEATURES) == 8, "Devem existir exatamente 8 features da água"
    assert "pH" in FEATURES, "A feature pH é obrigatória para este domínio"
    assert ALVO == "Pollution Level", "O alvo deve ser o nível de poluição"
