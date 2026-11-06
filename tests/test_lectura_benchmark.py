"""La recomendación no debe confundir ganar por casos con ganar por grupos."""
import pandas as pd
import pytest
from src.lectura_benchmark_regresion import adopta, contraste


def test_grupo_duplicado_puede_cambiar_el_signo_del_contraste():
    rows=[]
    for proc,errors in [("base",[5,5,5,5]),("nuevo",[0,0,6,7])]:
        for i,(fold,error) in enumerate(zip([1,1,2,3],errors)):
            rows.append(dict(protocolo="configuracion",Case_n=i,fold=fold,procedimiento=proc,
                             observado_N=10.0,predicho_N=10.0+error))
    scores=pd.DataFrame(rows)
    c=contraste(scores,"nuevo","base")
    assert c["delta_MAE_N"]==pytest.approx(-1.75)
    assert c["grupos_mejora"]==1 and c["grupos"]==3
    assert c["delta_al_omitir_grupo_max_N"]==pytest.approx(1.5)
    assert not adopta(c,1.0)
    with pytest.raises(ValueError,match="Cobertura"):
        contraste(scores.iloc[:-1],"nuevo","base")
