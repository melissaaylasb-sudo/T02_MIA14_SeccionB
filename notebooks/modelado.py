"""Cálculo reproducible del benchmark de regresión y sus diagnósticos.

Analisis conserva las particiones, los ajustes y las predicciones de cada etapa.
El cuaderno muestra las decisiones metodológicas y presenta sus resultados.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import product
from time import perf_counter
from typing import Any
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

MODELOS = ('SVR_RBF', 'XGBoost', 'CatBoost', 'Lineal_multiple', 'KNN', 'Arbol_decision')
PROTOCOLOS = ('configuracion', 'familia')
PROCEDIMIENTOS_AJUSTADOS = ('Tuning_rango_1', 'Tuning_rango_2', 'Selector_tuning')
Parametros = dict[str, Any]
Registros = dict[str, list[dict[str, Any]]]



class MediaPorNivel(RegressorMixin, BaseEstimator):
    """Media por nivel Y del entrenamiento. Para un nivel ausente usa la media global del entrenamiento."""

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.media_global_ = float(np.mean(y))
        self.medias_por_nivel_ = pd.Series(np.asarray(y)).groupby(X.numero_celdas_y.to_numpy()).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, 'medias_por_nivel_')
        return X.numero_celdas_y.map(self.medias_por_nivel_).fillna(self.media_global_).to_numpy(float)


class ValidacionGeometrica(TransformerMixin, BaseEstimator):
    """Rechaza respuestas, identificadores, infinitos y geometrías no positivas.

    Se admiten NaN en transform para imputación con estadísticas de entrenamiento.
    Una columna completamente ausente en entrenamiento no permite estimar mediana.
    """

    def __init__(self, columnas):
        self.columnas = columnas

    def _validar(self, geometrias):
        if not isinstance(geometrias, pd.DataFrame) or set(geometrias.columns) != set(self.columnas) or geometrias.columns.duplicated().any():
            raise ValueError('Se requieren exclusivamente las columnas geométricas autorizadas; sin respuesta ni identificadores.')
        x = geometrias[self.columnas].apply(pd.to_numeric, errors='raise').astype(float)
        a = x.to_numpy()
        if np.isinf(a).any() or (a <= 0).any():
            raise ValueError('La geometría debe ser positiva y finita, o NaN para imputación.')
        for col in ['numero_celdas_y', 'numero_total_pivotes']:
            v = x[col].dropna().to_numpy()
            if not np.equal(v, np.floor(v)).all():
                raise ValueError('Los conteos geométricos deben ser enteros.')
        return x

    def fit(self, X, y=None):
        x = self._validar(X)
        if x.isna().all().any():
            raise ValueError('Una entrada carece de observaciones en entrenamiento.')
        self.feature_names_in_ = np.array(self.columnas, dtype=object)
        self.n_features_in_ = len(self.columnas)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self._validar(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


class BaseIndependiente(TransformerMixin, BaseEstimator):
    """Base no supervisada de la matriz de diseño centrada, aprendida solo en fit."""

    def __init__(self, rtol: float=1e-10):
        self.rtol = rtol

    def fit(self, X: np.ndarray, y: np.ndarray | None=None):
        a = np.asarray(X, dtype=float)
        a = a - a.mean(axis=0)
        self.valores_singulares_ = np.linalg.svd(a, compute_uv=False)
        self.tol_ = float(self.rtol * self.valores_singulares_[0])
        elegidas = []
        for i in range(a.shape[1]):
            if np.linalg.matrix_rank(a[:, elegidas + [i]], tol=self.tol_) > len(elegidas):
                elegidas.append(i)
        if not elegidas:
            raise ValueError('Diseño OLS sin columnas informativas.')
        self.indices_ = np.asarray(elegidas, dtype=int)
        self.n_features_in_ = a.shape[1]
        self.rango_ = int(np.linalg.matrix_rank(a, tol=self.tol_))
        if self.rango_ != len(elegidas):
            raise ValueError('Base secuencial numéricamente ambigua; revisar la tolerancia.')
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'indices_')
        return np.asarray(X)[:, self.indices_]


def casos_de(texto):
    return [int(caso) for caso in str(texto).split(";")]


def calcular_metricas(cargas: np.ndarray, predicciones: np.ndarray) -> dict[str, float]:
    """Métricas en N; R² solo definido para al menos dos respuestas diferentes."""
    cargas = np.asarray(cargas, dtype=float)
    p = np.asarray(predicciones, dtype=float)
    if cargas.shape != p.shape or cargas.ndim != 1 or (not len(cargas)) or (not np.isfinite([cargas, p]).all()):
        raise ValueError('Respuestas/predicciones inválidas.')
    e = p - cargas
    a = np.abs(e)
    return {'MAE_N': float(a.mean()), 'RMSE_N': float(np.sqrt((e ** 2).mean())),
        'MedAE_N': float(np.median(a)),
        'R2_validacion': float(r2_score(cargas, p)) if len(cargas) > 1 and np.ptp(cargas) > 0 else np.nan, 'sesgo_N': float(e.mean())}


def grupos_nominales(datos, columnas):
    if datos[columnas].isna().any().any():
        raise ValueError('No se pueden verificar grupos nominales con claves ausentes.')
    return datos[columnas].apply(lambda fila: '|'.join((format(float(v), '.12g') for v in fila)), axis=1)


def _serializar(valor_dato: Any) -> str:
    return json.dumps(valor_dato, ensure_ascii=False, sort_keys=True, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def _identificadores(valores: Any) -> str:
    return ';'.join((str(int(v)) for v in valores))


def _predecir(estimador: Any, geometrias: pd.DataFrame) -> tuple[np.ndarray, float]:
    inicio = perf_counter()
    p = np.asarray(estimador.predict(geometrias), dtype=float).reshape(-1)
    segundos = perf_counter() - inicio
    if len(p) != len(geometrias) or not np.isfinite(p).all():
        raise ValueError('Predicción no finita o de longitud incorrecta.')
    return (p, segundos)


def guardar_ajuste(familia, parametros, geometrias, contexto, registros, estimador, segundos, avisos=""):
    registro = {**contexto, 'familia_real': familia, 'n_entrenamiento': len(geometrias), 'casos_entrenamiento': _identificadores(geometrias.index),
        'parametros_json': _serializar(parametros), 'tiempo_ajuste_s': segundos,
        'avisos': avisos}
    for paso, atributo in [('imputacion', 'statistics_'), ('escala', 'mean_'), ('escala', 'scale_')]:
        if paso in estimador.named_steps:
            registro[paso + atributo] = _serializar(getattr(estimador.named_steps[paso], atributo).tolist())
    registros['ajustes'].append(registro)


def metricas_internas(cargas: np.ndarray, predicciones: np.ndarray, grupos: np.ndarray) -> tuple[float, float]:
    """MAE conjunto por probeta y MAD no escalada de las MAE por grupo."""
    errores = np.abs(np.asarray(cargas) - np.asarray(predicciones))
    medias = pd.Series(errores).groupby(np.asarray(grupos)).mean()
    return (float(errores.mean()), float(np.median(np.abs(medias - np.median(medias)))))


def ordenar_familias(puntuaciones: dict[str, tuple[float, float]], configuracion: Parametros) -> list[str]:
    """Ordena por banda de MAE, estabilidad interna y orden congelado."""
    restantes = dict(puntuaciones)
    orden = []
    while restantes:
        limite = min((v[0] for v in restantes.values())) + configuracion['banda_cribado_N']
        admisibles = [f for f, v in restantes.items() if v[0] <= limite + 1e-12]
        indice_elegido = min(admisibles,
            key=lambda f: (restantes[f][1], restantes[f][0], configuracion['orden_complejidad'].index(f), f))
        orden.append(indice_elegido)
        del restantes[indice_elegido]
    return orden


def registrar_reservas_internas(geometrias, cargas, grupos, particiones_por_grupo, contexto, registros):
    if not geometrias.index.equals(cargas.index) or len(grupos) != len(geometrias):
        raise ValueError('Índices internos incongruentes.')
    verificados = []
    for s, (ind_entrenamiento, ind_validacion) in enumerate(particiones_por_grupo, 1):
        if set(ind_entrenamiento) & set(ind_validacion) or set(grupos[ind_entrenamiento]) & set(grupos[ind_validacion]) or set(ind_entrenamiento) | set(ind_validacion) != set(range(len(geometrias))):
            raise ValueError('Fuga en partición interna.')
        verificados.extend(ind_validacion)
        registros['particiones_internas'].append({**contexto, 'particion_interna': s, 'casos_entrenamiento': _identificadores(geometrias.index[ind_entrenamiento]),
            'casos_validacion': _identificadores(geometrias.index[ind_validacion]), 'grupos_entrenamiento': _serializar(sorted(set(grupos[ind_entrenamiento]))),
            'grupos_validacion': _serializar(sorted(set(grupos[ind_validacion])))})
    if sorted(verificados) != list(range(len(geometrias))):
        raise ValueError('Cobertura interna OOF incompleta.')


def indice_preferido(errores_internos: list[tuple[float, float]]) -> int:
    """Menor MAE; empate numérico por MAD y orden del plan."""
    minimo_interno = min((x[0] for x in errores_internos))
    return min((i for i, v in enumerate(errores_internos) if v[0] <= minimo_interno + 1e-10), key=lambda i: (errores_internos[i][1], i))


def contraste(predicciones_externas: pd.DataFrame, candidata: str, base: str) -> dict[str, Any]:
    """Errores emparejados y sensibilidad al grupo en un solo protocolo."""
    if predicciones_externas.protocolo.nunique()!=1:
        raise ValueError("Comparar un protocolo por vez")
    columnas=["id_probeta","pliegue","observado_N","predicho_N"]
    a=predicciones_externas[predicciones_externas.procedimiento.eq(candidata)][columnas].copy()
    b=predicciones_externas[predicciones_externas.procedimiento.eq(base)][columnas].copy()
    p=a.merge(b,on=["id_probeta","pliegue"],suffixes=("_a","_b"),validate="one_to_one")
    if len(p)!=len(a) or len(p)!=len(b) or not len(p):
        raise ValueError("Cobertura OOF desigual")
    cargas=p.observado_N_a.to_numpy()
    if not np.allclose(cargas,p.observado_N_b,atol=1e-12,rtol=0):
        raise ValueError("Respuesta no comparable")
    ea = np.abs(p.predicho_N_a - cargas)
    eb = np.abs(p.predicho_N_b - cargas)
    d=pd.Series(ea-eb,index=p.index)
    gm=d.groupby(p.pliegue).mean()
    al_omitir=np.array([d[p.pliegue.ne(f)].mean() for f in gm.index])
    return {"delta_MAE_N":float(d.mean()),"delta_RMSE_N":float(np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2))),
            "grupos_mejora":int((gm < -1e-10).sum()),"grupos_empate":int((gm.abs()<=1e-10).sum()),
            "grupos":len(gm),"fraccion_grupos_mejora":float((gm < -1e-10).mean()),
            "delta_al_omitir_grupo_min_N":float(al_omitir.min()),"delta_al_omitir_grupo_max_N":float(al_omitir.max())}


def adopta(c: dict[str,Any], margen: float) -> bool:
    return (c["delta_MAE_N"] <= -margen+1e-10 and c["delta_RMSE_N"] <= 1e-10 and
            c["fraccion_grupos_mejora"]>0.5 and c["delta_al_omitir_grupo_max_N"] < -1e-10)


def lectura_en_memoria(tablas, receta, configuracion, politica):
    """Distingue menor MAE, recomendación registrada y alternativas no lineales."""
    predicciones_externas = tablas["predicciones_validacion"]
    tabla_metricas = tablas["metricas_resumen"]
    nominal = predicciones_externas[predicciones_externas.protocolo.eq("configuracion")]
    m = tabla_metricas[tabla_metricas.protocolo.eq("configuracion")].set_index("procedimiento")
    f = tabla_metricas[tabla_metricas.protocolo.eq("familia")].set_index("procedimiento")
    modelos = politica["modelos_iniciales"]
    banda = politica["banda_descriptiva_N"]
    minimo = m.loc[modelos, "MAE_N"].min()
    fijo = next(x for x in politica["prioridad_parsimonia_en_banda"] if m.loc[x,"MAE_N"] <= minimo+banda+1e-10)
    comparacion = contraste(nominal, "Selector_tuning", fijo)
    principal = "Selector_tuning" if adopta(comparacion, banda) else fijo
    frente_a_media = contraste(nominal, principal, "Media_por_nivel")
    operativa = principal if adopta(frente_a_media, banda) else "Media_por_nivel"
    es_selector = principal == "Selector_tuning"
    familia = receta["selector"]["familia_real"] if es_selector else principal
    parametros = receta["selector"]["parametros"] if es_selector else configuracion["parametros_iniciales"][principal]
    respaldos = [x for x in m.loc[modelos].sort_values("MAE_N").index if x != familia][:2]
    estrategias = modelos + ["Tuning_rango_1","Tuning_rango_2","Selector_tuning"]
    categorias = {"lineal":"Lineal_multiple","distancias":"KNN",
        "arboles":m.loc[["Arbol_decision","XGBoost","CatBoost"],"MAE_N"].idxmin(),
        "grupo_A_literatura":m.loc[["SVR_RBF","XGBoost","CatBoost"],"MAE_N"].idxmin(),
        "menor_MAE_inicial":m.loc[modelos,"MAE_N"].idxmin(),
        "menor_DE_entre_iniciales":m.loc[modelos,"DE_MAE_pliegue_N"].idxmin(),
        "menor_MAE_procedimientos_con_ajuste":m.loc[estrategias,"MAE_N"].idxmin(),
        "menor_MAE_toda_tabla_incluidos_controles":m.MAE_N.idxmin(),"compromiso":principal}
    return {"procedimiento_principal":principal, "familia_principal":familia,
        "parametros_principal":parametros,"respaldos":respaldos,"referencia_operativa":operativa,
        "motivo":"Política de comparación declarada: banda de MAE, dispersión y contraste emparejado por grupo.",
        "MAE_principal_N":float(m.loc[principal,"MAE_N"]), "MAE_media_nivel_N":float(m.loc["Media_por_nivel","MAE_N"]),
        "MAE_referencia_familia_excluida_N":float(f.loc[operativa,"MAE_N"]),
        "MAE_principal_familia_excluida_N":float(f.loc[principal,"MAE_N"]),
        "familia_con_menor_MAE_transferencia":f.loc[modelos,"MAE_N"].idxmin(), "categorias":categorias,
        "procedimiento_fijo_preferido":fijo,"selector_vs_fijo":comparacion,"principal_vs_media_nivel":frente_a_media}


class Analisis:
    """Estado y etapas del análisis sobre una campaña verificada."""

    def __init__(self, campana, *, semilla_busqueda=20261106, semilla_modelos=42):
        """Preparar geometría, cargas y particiones de la campaña verificada."""
        self.muestras = campana.muestras
        self.ficha_variables = campana.ficha_variables
        self.particiones = campana.particiones
        self.configuracion = deepcopy(campana.configuracion)
        self.configuracion['semilla'] = semilla_busqueda
        parametros = self.configuracion['parametros_iniciales']
        parametros['XGBoost']['random_state'] = semilla_modelos
        parametros['CatBoost']['random_seed'] = semilla_modelos
        parametros['Arbol_decision']['random_state'] = semilla_modelos
        self.politica = campana.politica
        self.predictores = list(campana.predictores)
        self.objetivo = campana.objetivo
        self.grupo_cols = list(campana.grupo_cols)
        self.modelos_iniciales = tuple(self.configuracion['familias'])
        self.referencias = tuple(self.configuracion['referencias'])
        self.unidades = dict(zip(self.ficha_variables.variable, self.ficha_variables.Unidad))
        self.presupuesto = self.configuracion['presupuesto_por_candidata']
        assert self.modelos_iniciales == MODELOS
        assert len(self.predictores) == len(set(self.predictores)) == 10
        self._etapas_completadas = set()
        self._etapa_pendiente = None
        self.datos = self.muestras.copy().sort_values("id_probeta").reset_index(drop=True)
        assert "id_probeta" in self.datos and self.datos.id_probeta.is_unique and not self.datos.id_probeta.isna().any()
        self.datos = self.datos.loc[self.datos[self.objetivo].notna()].reset_index(drop=True)
        assert np.isfinite(self.datos[self.objetivo]).all() and (self.datos[self.objetivo] > 0).all()
        self.geometrias = self.datos.set_index("id_probeta")[self.predictores]
        self.cargas = self.datos.set_index("id_probeta")[self.objetivo]
        self.nominal = grupos_nominales(self.datos, self.grupo_cols).to_numpy()
        self.indices = {int(identificador): i for i, identificador in enumerate(self.geometrias.index)}
        self.particiones_por_grupo = {}
        for protocolo in self.configuracion["protocolos"]:
            self.tabla_pliegues = self.particiones.loc[self.particiones.protocolo == protocolo].sort_values("pliegue")
            assert self.tabla_pliegues.pliegue.tolist() == list(range(1, len(self.tabla_pliegues) + 1))
            self.particiones_por_grupo[protocolo] = [(self.indices_casos(r.casos_entrenamiento), self.indices_casos(r.casos_prueba)) for r in self.tabla_pliegues.itertuples()]
            grupos = self.nominal if protocolo == "configuracion" else self.geometrias.numero_celdas_y.to_numpy()
            evaluados = []
            for ind_entrenamiento, ind_prueba in self.particiones_por_grupo[protocolo]:
                assert not set(ind_entrenamiento) & set(ind_prueba) and set(ind_entrenamiento) | set(ind_prueba) == set(range(len(self.geometrias)))
                assert not set(grupos[ind_entrenamiento]) & set(grupos[ind_prueba]) and len(set(grupos[ind_prueba])) == 1
                evaluados.extend(ind_prueba)
            assert sorted(evaluados) == list(range(len(self.geometrias)))
        assert len(self.particiones_por_grupo["configuracion"]) == len(np.unique(self.nominal))
        assert len(self.particiones_por_grupo["familia"]) == self.geometrias.numero_celdas_y.nunique()

    def _iniciar_etapa(self, nombre, requisito=None):
        if self._etapa_pendiente is not None:
            raise RuntimeError(
                f'La etapa {self._etapa_pendiente} quedó incompleta. '
                'Crea una nueva instancia de Analisis y ejecuta las etapas en orden.'
            )
        if nombre in self._etapas_completadas:
            return False
        if requisito is not None and requisito not in self._etapas_completadas:
            raise RuntimeError(f'Ejecuta {requisito} antes de {nombre}.')
        self._etapa_pendiente = nombre
        return True

    def _terminar_etapa(self, nombre):
        self._etapas_completadas.add(nombre)
        self._etapa_pendiente = None

    def referencia_inicial(self, familia):
        parametros = self.configuracion["parametros_iniciales"][familia]
        if familia == "Dummy_mediana":
            return Pipeline([("regresor", DummyRegressor(**parametros))])
        if familia == "Media_por_nivel":
            return Pipeline([("regresor", MediaPorNivel())])
        if familia != "Ridge_historico":
            raise ValueError(f"Referencia inesperada: {familia}")
        return Pipeline([
            ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("varianza", VarianceThreshold()),
            ("escala", StandardScaler()),
            ("regresor", Ridge(**parametros)),
        ])

    def indices_casos(self, valores):
        if isinstance(valores, str):
            valores = valores.split(";")
        return np.array([self.indices[int(identificador)] for identificador in valores], dtype=int)

    def preparar_preprocesamiento(self, familia, configuracion):
        """Reutilizar las transformaciones geométricas del modelo especificado."""
        pasos = [
            ("geometria", ValidacionGeometrica(self.predictores)),
            ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
            ("varianza", VarianceThreshold()),
        ]
        if familia in ("SVR_RBF", "KNN", "Lineal_multiple"):
            pasos.append(("escala", StandardScaler()))
        if familia == "Lineal_multiple":
            pasos.append(("base", BaseIndependiente(configuracion["ols"]["tolerancia_svd_relativa"])))
        return pasos

    def construir_modelo(self, familia: str, parametros: Parametros, configuracion: Parametros) -> Pipeline:
        if familia in self.referencias:
            return self.referencia_inicial(familia)
        constructores = {
            "SVR_RBF": SVR, "XGBoost": XGBRegressor, "CatBoost": CatBoostRegressor,
            "Lineal_multiple": LinearRegression, "KNN": KNeighborsRegressor,
            "Arbol_decision": DecisionTreeRegressor,
        }
        pasos = self.preparar_preprocesamiento(familia, configuracion)
        return Pipeline([*pasos, ("regresor", constructores[familia](**parametros))])

    def nombres_sin_constantes(self, modelo):
        return np.array(self.predictores)[modelo.named_steps["varianza"].get_support()]

    def _etapa(self, proc: str) -> str:
        return 'inicial' if proc in self.modelos_iniciales else 'referencia' if proc in self.referencias else 'optimizada'

    def _grupo(self, proc: str) -> str:
        if proc in self.referencias:
            return 'referencia'
        if proc in PROCEDIMIENTOS_AJUSTADOS:
            return 'procedimiento_adaptativo'
        return 'A_materiales' if proc in self.modelos_iniciales[:3] else 'B_clasicos'

    def _ajustar(self, familia: str, parametros: Parametros, configuracion: Parametros, geometrias: pd.DataFrame, cargas: pd.Series, contexto: Parametros,
        registros: Registros) -> tuple[Any, float]:
        if not geometrias.index.equals(cargas.index) or not geometrias.index.is_unique or set(geometrias) != set(self.predictores):
            raise ValueError('Contrato de entrenamiento inválido.')
        if familia == 'KNN' and parametros['n_neighbors'] > len(cargas):
            raise ValueError('k supera el tamaño del entrenamiento del pliegue.')
        estimador = self.construir_modelo(familia, parametros, configuracion)
        inicio = perf_counter()
        with warnings.catch_warnings(record=True) as avisos_capturados:
            warnings.simplefilter('always')
            estimador.fit(geometrias, cargas)
        segundos = perf_counter() - inicio
        guardar_ajuste(familia, parametros, geometrias, contexto, registros, estimador, segundos,
                       ' | '.join(sorted({str(w.message) for w in avisos_capturados})))
        return estimador, segundos

    def evaluar_reserva(self, familia, parametros, geometria_entrenamiento, cargas_entrenamiento, geometria_prueba, contexto, configuracion, registros):
        """Ajustar en entrenamiento y cronometrar la predicción de la reserva por separado."""
        modelo, t_ajuste = self._ajustar(familia, parametros, configuracion, geometria_entrenamiento, cargas_entrenamiento, contexto, registros)
        prediccion_prueba, segundos_prediccion = _predecir(modelo, geometria_prueba)
        prediccion_entrenamiento, _ = _predecir(modelo, geometria_entrenamiento)
        return modelo, t_ajuste, prediccion_prueba, segundos_prediccion, prediccion_entrenamiento

    def _registrar_predicciones(self, proc: str, familia: str, contexto_etapa: Parametros, geometrias: pd.DataFrame, cargas: pd.Series, entrenamiento: np.ndarray,
        prueba: np.ndarray, nominal: np.ndarray, p: np.ndarray, predicciones_entrenamiento: np.ndarray, segundos_ajuste: float, segundos_prediccion: float,
        segundos_busqueda: float, prediccion_inicial: np.ndarray, registros: Registros) -> None:
        contexto_registro = {**contexto_etapa, 'procedimiento': proc, 'etapa': self._etapa(proc), 'familia_real': familia}
        observaciones_prueba = cargas.iloc[prueba].to_numpy()
        met = calcular_metricas(observaciones_prueba, p)
        registros['metricas_pliegues'].append({**contexto_registro, 'n_entrenamiento': len(entrenamiento), 'n_prueba': len(prueba),
            'MAE_entrenamiento_N': float(np.mean(np.abs(cargas.iloc[entrenamiento].to_numpy() - predicciones_entrenamiento))), 'MAE_validacion_N': met['MAE_N'],
            'RMSE_validacion_N': met['RMSE_N'], 'MedAE_validacion_N': met['MedAE_N'],
            'sesgo_validacion_N': met['sesgo_N'], 'tiempo_ajuste_s': segundos_ajuste, 'tiempo_prediccion_s': segundos_prediccion,
            'tiempo_busqueda_s': segundos_busqueda})
        for j, obs, predicciones, prediccion_original in zip(prueba, observaciones_prueba, p, prediccion_inicial):
            registros['predicciones_validacion'].append({**contexto_registro, 'id_probeta': int(geometrias.index[j]),
                'numero_celdas_y': int(geometrias.iloc[j].numero_celdas_y), 'grupo_nominal': nominal[j], 'observado_N': obs,
                'predicho_N': predicciones, 'residuo_N': obs - predicciones, 'error_absoluto_N': abs(obs - predicciones),
                'predicho_inicial_misma_familia_N': prediccion_original})
        for j, predicciones in zip(entrenamiento, predicciones_entrenamiento):
            registros['predicciones_entrenamiento'].append({**contexto_registro, 'id_probeta': int(geometrias.index[j]), 'observado_N': cargas.iloc[j],
                'predicho_N': predicciones})

    def ecuacion_en_unidades_originales(self, estimador):
        """Deshace el escalado de la base OLS y conserva las unidades geométricas."""
        indice = estimador.named_steps["base"].indices_
        escala = estimador.named_steps["escala"]
        modelo = estimador.named_steps["regresor"]
        coeficientes = pd.Series(modelo.coef_ / escala.scale_[indice],
                                 index=self.nombres_sin_constantes(estimador)[indice])
        intercepto = float(modelo.intercept_ - coeficientes.to_numpy() @ escala.mean_[indice])
        return intercepto, coeficientes

    def _diagnosticar(self, estimador: Any, familia: str, geometrias: pd.DataFrame, geometria_reservada: pd.DataFrame, contexto_registro: Parametros, registros: Registros) -> None:
        """Diagnósticos descriptivos del modelo ajustado; nunca intervienen en el cribado."""
        m = estimador.named_steps['regresor']
        registro = {**contexto_registro, 'familia_real': familia, 'n_entrenamiento': len(geometrias)}
        if 'varianza' in estimador.named_steps:
            nombres_variables = list(self.nombres_sin_constantes(estimador))
            registro['n_variables_tras_varianza'] = len(nombres_variables)
        if familia == 'Lineal_multiple':
            base_independiente = estimador.named_steps['base']
            singulares = base_independiente.valores_singulares_
            z = estimador[:-2].transform(geometrias)
            zb = base_independiente.transform(z)
            s = np.linalg.svd(zb - zb.mean(axis=0), compute_uv=False)
            ordenada_origen, pendientes = self.ecuacion_en_unidades_originales(estimador)
            conservadas, pendientes_fisicas = pendientes.index.to_list(), pendientes.to_numpy()
            registro.update(rango_centrado=base_independiente.rango_, rango_base=len(conservadas),
                condicion_centrada=float('inf') if base_independiente.rango_ < len(nombres_variables) else singulares[0] / singulares[-1],
                condicion_base=s[0] / s[-1], variables_base=';'.join(conservadas), intercepto_fisico_N=ordenada_origen,
                tol_svd=base_independiente.tol_, singulares_centrados_json=_serializar(singulares.tolist()))
            for c in self.predictores:
                i = conservadas.index(c) if c in conservadas else None
                registros['coeficientes_ols'].append({**contexto_registro, 'variable': c, 'unidad_variable': self.unidades[c],
                    'estado': 'retenida' if i is not None else 'redundante_en_entrenamiento' if c in nombres_variables else 'constante_en_entrenamiento', 'coeficiente_N_por_unidad': pendientes_fisicas[i] if i is not None else np.nan, 'coeficiente_estandarizado_N': m.coef_[i] if i is not None else np.nan, 'intercepto_fisico_N': ordenada_origen})
            np.testing.assert_allclose(ordenada_origen + estimador[:2].transform(geometrias)[conservadas].to_numpy() @ pendientes_fisicas, estimador.predict(geometrias), atol=1e-07, rtol=1e-10)
        elif familia == 'SVR_RBF':
            registro.update(n_vectores_soporte=len(m.support_), fraccion_vectores_soporte=len(m.support_) / len(geometrias),
                casos_soporte=_identificadores(geometrias.index[m.support_]))
        elif familia == 'KNN':
            distancias, indice_seleccionado = m.kneighbors(estimador[:-1].transform(geometria_reservada))
            for identificador, ds, ix in zip(geometria_reservada.index, distancias, indice_seleccionado):
                for i, (d, j) in enumerate(zip(ds, ix), 1):
                    registros['vecinos_knn'].append({**contexto_registro, 'id_probeta': int(identificador), 'orden_vecino': i,
                        'vecino_entrenamiento': int(geometrias.index[j]), 'distancia_escalada': float(d), 'p': m.p,
                        'weights': m.weights})
            registro.update(k=m.n_neighbors, distancia_media_vecinos=float(distancias.mean()))
        elif familia == 'Arbol_decision':
            registro.update(profundidad=m.get_depth(), hojas=m.get_n_leaves())
            registros['reglas_arbol'].append({**contexto_registro,
                'regla_descriptiva_entrenamiento': export_text(m, feature_names=nombres_variables, decimals=6)})
        elif familia == 'Media_por_nivel':
            registro.update(n_pruebas_con_media_global=int((~geometria_reservada.numero_celdas_y.isin(m.medias_por_nivel_)).sum()), media_global_entrenamiento_N=m.media_global_)
        registros['diagnosticos_modelos'].append(registro)

    def _resumir_metricas(self, registros: Registros) -> None:
        predicciones_externas = pd.DataFrame(registros["predicciones_validacion"])
        tabla_pliegues = pd.DataFrame(registros["metricas_pliegues"])
        claves = ["protocolo", "procedimiento"]
        grupos = predicciones_externas.groupby(claves, sort=False)
        por_pliegue = tabla_pliegues.groupby(claves, sort=False)
        controles = {k: tabla.error_absoluto_N.mean() for k, tabla in grupos if k[1] in self.referencias}
        variables_referencia = {self.referencias[0]: "sin_geometria", self.referencias[1]: "numero_celdas_y"}
        sufijos = dict(zip(self.referencias, ("dummy", "media_nivel", "Ridge")))
        for (protocolo, proc), pruebas in grupos:
            por_reserva = por_pliegue.get_group((protocolo, proc))
            mae_pliegue = por_reserva.MAE_validacion_N
            q25, q75 = mae_pliegue.quantile([.25, .75])
            resumen = calcular_metricas(pruebas.observado_N.to_numpy(), pruebas.predicho_N.to_numpy())
            resumen.update({
                "protocolo": protocolo, "procedimiento": proc, "etapa": self._etapa(proc), "grupo": self._grupo(proc),
                "conjunto_variables": variables_referencia.get(proc, "A"), "n": len(pruebas), "n_pliegues": len(por_reserva),
                "DE_error_absoluto_N": pruebas.error_absoluto_N.std(), "DE_MAE_pliegue_N": mae_pliegue.std(),
                "RIC_MAE_pliegue_N": q75 - q25, "MAE_entrenamiento_medio_N": por_reserva.MAE_entrenamiento_N.mean(),
            })
            for columna in ("tiempo_ajuste_s", "tiempo_prediccion_s", "tiempo_busqueda_s"):
                resumen[columna] = por_reserva[columna].sum()
            for referencia_base, sufijo in sufijos.items():
                resumen[f"delta_MAE_vs_{sufijo}_N"] = resumen["MAE_N"] - controles[protocolo, referencia_base]
            mae_inicial = (pruebas.observado_N - pruebas.predicho_inicial_misma_familia_N).abs().mean()
            resumen["delta_MAE_vs_inicial_misma_familia_N"] = resumen["MAE_N"] - mae_inicial
            registros["metricas_resumen"].append(resumen)

    def resumen(self, proto, procedimientos=MODELOS):
        filas = self.metricas[self.metricas.protocolo.eq(proto)].set_index("procedimiento")
        return filas.loc[list(procedimientos)]

    def valor(self, proto, procedimiento, campo="MAE_N"):
        return float(self.resumen(proto, (procedimiento,)).iloc[0][campo])

    def datos_internos(self, entrenamiento, protocolo):
        """Prepara una sola vez los grupos y las reservas que comparten el cribado y la búsqueda."""
        geometria_entrenamiento, cargas_entrenamiento = self.geometrias.iloc[entrenamiento], self.cargas.iloc[entrenamiento]
        if protocolo == "configuracion":
            grupos = self.nominal[entrenamiento]
            separador = GroupKFold(n_splits=5)
            etiquetas = grupos
        elif protocolo == "familia":
            grupos = geometria_entrenamiento.numero_celdas_y.to_numpy()
            separador = LeaveOneGroupOut()
            etiquetas = grupos.astype(int).astype(str)
        else:
            raise ValueError(f"Protocolo desconocido: {protocolo}")
        pliegues = list(separador.split(geometria_entrenamiento, groups=etiquetas))
        return geometria_entrenamiento, cargas_entrenamiento, grupos, pliegues

    def evaluar_internamente(self, familia, orden, parametros, geometrias, cargas, grupos, particiones_por_grupo, configuracion, contexto, registros) -> tuple[tuple[float, float], float]:
        inicio = perf_counter()
        p = np.full(len(geometrias), np.nan)
        identificadores, medido = geometrias.index.to_numpy(), cargas.to_numpy()
        id_receta = f'{familia}_{orden:02d}'
        for s, (ind_entrenamiento, ind_validacion) in enumerate(particiones_por_grupo, 1):
            c = {**contexto, 'particion_interna': s, 'etapa_ajuste': 'cribado' if orden == 0 else 'busqueda', 'id_configuracion': id_receta}
            estimador, _ = self._ajustar(familia, parametros, configuracion, geometrias.iloc[ind_entrenamiento], cargas.iloc[ind_entrenamiento], c, registros)
            p[ind_validacion], _ = _predecir(estimador, geometrias.iloc[ind_validacion])
            for j in ind_validacion:
                registros['predicciones_internas'].append({**c, 'familia_real': familia, 'id_probeta': int(identificadores[j]),
                    'grupo_interno': grupos[j], 'observado_N': float(medido[j]), 'predicho_N': float(p[j])})
        return (metricas_internas(medido, p, grupos), perf_counter() - inicio)

    def cribar_entrenamiento(self, geometrias, cargas, grupos, particiones_por_grupo, configuracion, contexto, registros):
        inicio_medicion = perf_counter()
        registrar_reservas_internas(geometrias, cargas, grupos, particiones_por_grupo, contexto, registros)
        familias = [f for f in self.modelos_iniciales if f in configuracion["espacios"]]
        cribado_familias = {f: self.evaluar_internamente(f, 0, configuracion["parametros_iniciales"][f],
                  geometrias, cargas, grupos, particiones_por_grupo, configuracion, contexto, registros) for f in familias}
        clasificacion = ordenar_familias({f: v[0] for f, v in cribado_familias.items()}, configuracion)
        preseleccionadas = clasificacion[:configuracion['numero_familias_tuning']]
        for puesto_cribado, f in enumerate(clasificacion, 1):
            (mae, mad), segundos = cribado_familias[f]
            registros['cribado_interno'].append({**contexto, 'familia_real': f, 'MAE_interno_N': mae,
                'MAD_grupos_N': mad, 'rango_cribado': puesto_cribado, 'elegible_ajuste': True,
                'seleccionada_ajuste': f in preseleccionadas, 'tiempo_s': segundos})
        return cribado_familias, preseleccionadas, perf_counter() - inicio_medicion

    def fijar_candidatas(self, configuracion: Parametros) -> dict[str, list[Parametros]]:
        """Muestrea una vez cada pequeño espacio, sin consultar cargas, pliegues ni resultados."""
        resultado_busqueda = {}
        for i, familia in enumerate(self.modelos_iniciales):
            if familia not in configuracion['espacios']:
                continue
            receta_fija = configuracion['parametros_iniciales'][familia]
            espacio = configuracion['espacios'][familia]
            claves_ordenadas = list(self.orden_parametros[familia])
            combinaciones = [dict(receta_fija, **dict(zip(claves_ordenadas, fila))) for fila in product(*(espacio[k] for k in claves_ordenadas))]
            combinaciones = [p for p in combinaciones if p != receta_fija]
            generador = np.random.default_rng(configuracion['semilla'] + i)
            indices_elegidos = generador.choice(len(combinaciones), size=configuracion['presupuesto_por_candidata'] - 1, replace=False)
            resultado_busqueda[familia] = [receta_fija.copy(), *[combinaciones[j] for j in indices_elegidos]]
        return resultado_busqueda

    def afinar(self, geometrias, cargas, grupos, particiones_por_grupo, configuracion, planes_busqueda, contexto, registros, cribado_previo):
        cribado_familias, preseleccionadas, segundos_cribado = cribado_previo
        inicio_medicion = perf_counter()
        ganadoras_ajuste = []
        for puesto_cribado, f in enumerate(preseleccionadas, 1):
            valores = [cribado_familias[f]] + [self.evaluar_internamente(f, i, planes_busqueda[f][i], geometrias, cargas, grupos, particiones_por_grupo, configuracion, contexto, registros) for i in range(1, len(planes_busqueda[f]))]
            indice_seleccionado = indice_preferido([s for s, _ in valores])
            mejor_valor = np.inf
            for i, ((mae, mad), segundos) in enumerate(valores):
                mejor_valor = min(mae, mejor_valor)
                registros['busqueda_interna'].append({**contexto, 'familia_real': f, 'orden': i + 1,
                    'id_configuracion': f'{f}_{i:02d}', 'parametros_json': _serializar(planes_busqueda[f][i]), 'MAE_interno_N': mae,
                    'MAD_grupos_N': mad, 'mejor_acumulado_N': mejor_valor, 'seleccionada': i == indice_seleccionado,
                    'tiempo_s': segundos if i else 0.0, 'reuso_cribado': i == 0})
            (mae, mad), _ = valores[indice_seleccionado]
            ganadoras_ajuste.append({'rango_cribado': puesto_cribado, 'familia_real': f, 'id_configuracion': f'{f}_{indice_seleccionado:02d}',
                'parametros': planes_busqueda[f][indice_seleccionado], 'MAE_interno_N': mae, 'MAD_grupos_N': mad})
        indice_elegido = indice_preferido([(w['MAE_interno_N'], w['MAD_grupos_N']) for w in ganadoras_ajuste])
        for i, w in enumerate(ganadoras_ajuste):
            registros['seleccion_ajuste'].append({**contexto, **{k: v for k, v in w.items() if k != 'parametros'},
                'parametros_json': _serializar(w['parametros']), 'elegida_selector': i == indice_elegido})
        return (ganadoras_ajuste, indice_elegido, segundos_cribado + perf_counter() - inicio_medicion)

    def evaluar_referencias(self):
        """Calcular las referencias por reserva y conservar sus ajustes para reutilizarlos."""
        if not self._iniciar_etapa('evaluar_referencias', None):
            return

        casos = self.muestras.set_index("id_probeta").sort_index()
        geometria_base, cargas_base = casos[self.predictores], casos[self.objetivo]
        self.ajustes_base, self.predicciones_base = {}, []
        with threadpool_limits(limits=1):
            for reserva in self.particiones.itertuples():
                entrenamiento, prueba = casos_de(reserva.casos_entrenamiento), casos_de(reserva.casos_prueba)
                assert set(entrenamiento).isdisjoint(prueba)
                for familia in self.configuracion["referencias"]:
                    modelo = self.referencia_inicial(familia)
                    inicio = perf_counter()
                    modelo.fit(geometria_base.loc[entrenamiento], cargas_base.loc[entrenamiento])
                    segundos_ajuste = perf_counter() - inicio
                    inicio = perf_counter()
                    predicciones = modelo.predict(geometria_base.loc[prueba])
                    segundos_prediccion = perf_counter() - inicio
                    clave = (reserva.protocolo, reserva.pliegue, familia)
                    self.ajustes_base[clave] = (modelo, segundos_ajuste, predicciones, segundos_prediccion, modelo.predict(geometria_base.loc[entrenamiento]))
                    for caso, valor_predicho in zip(prueba, predicciones):
                        self.predicciones_base.append((reserva.protocolo, familia, caso, cargas_base.at[caso], valor_predicho))

        self.predicciones_referencia = pd.DataFrame(self.predicciones_base, columns=["protocolo", "referencia", "caso", "observado_N", "predicho_N"])
        grupos_referencia = self.predicciones_referencia.groupby(["protocolo", "referencia"])
        assert grupos_referencia.size().eq(len(casos)).all()
        assert not self.predicciones_referencia.duplicated(["protocolo", "referencia", "caso"]).any()
        self.metricas_referencia = pd.DataFrame([
            {"protocolo": protocolo, "procedimiento": referencia,
             **calcular_metricas(grupo.observado_N, grupo.predicho_N)}
            for (protocolo, referencia), grupo in grupos_referencia
        ])
        self._terminar_etapa('evaluar_referencias')

    def evaluar_iniciales(self):
        """Evaluar los seis modelos y registrar los resultados de las referencias."""
        if not self._iniciar_etapa('evaluar_iniciales', 'evaluar_referencias'):
            return

        self.registros: Registros = defaultdict(list)
        self.predicciones_guardadas = {}
        self.inicio_medicion = perf_counter()
        with threadpool_limits(limits=1):
            for protocolo, reservas in self.particiones_por_grupo.items():
                for pliegue, (ind_entrenamiento, ind_prueba) in enumerate(reservas, 1):
                    contexto = {"protocolo": protocolo, "pliegue": pliegue}
                    self.registros["particiones_externas"].append({
                        **contexto, "casos_entrenamiento": _identificadores(self.geometrias.index[ind_entrenamiento]), "casos_prueba": _identificadores(self.geometrias.index[ind_prueba])
                    })
                    geometria_entrenamiento, cargas_entrenamiento, geometria_prueba = self.geometrias.iloc[ind_entrenamiento], self.cargas.iloc[ind_entrenamiento], self.geometrias.iloc[ind_prueba]
                    for familia in (*self.modelos_iniciales, *self.referencias):
                        parametros = self.configuracion["parametros_iniciales"][familia]
                        ajuste = {**contexto, "etapa_ajuste": "externa_inicial", "id_configuracion": familia + "_00"}
                        clave = (protocolo, pliegue, familia)
                        if familia in self.referencias:
                            estimador, segundos_ajuste, predicciones, segundos_prediccion, predicho_entrenamiento = self.ajustes_base[clave]
                            guardar_ajuste(familia, parametros, geometria_entrenamiento, ajuste, self.registros, estimador, segundos_ajuste)
                        else:
                            estimador, segundos_ajuste, predicciones, segundos_prediccion, predicho_entrenamiento = self.evaluar_reserva(
                                familia, parametros, geometria_entrenamiento, cargas_entrenamiento, geometria_prueba, ajuste, self.configuracion, self.registros)
                        self.predicciones_guardadas[clave] = predicciones
                        self._registrar_predicciones(familia, familia, contexto, self.geometrias, self.cargas, ind_entrenamiento, ind_prueba, self.nominal, predicciones, predicho_entrenamiento, segundos_ajuste, segundos_prediccion, 0, predicciones, self.registros)
                        self._diagnosticar(estimador, familia, geometria_entrenamiento, geometria_prueba, {**contexto, "procedimiento": familia}, self.registros)

        self._resumir_metricas(self.registros)
        self.metricas = pd.DataFrame(self.registros["metricas_resumen"])
        self.tabla_pliegues = pd.DataFrame(self.registros["metricas_pliegues"])
        self.predicciones_externas = pd.DataFrame(self.registros["predicciones_validacion"])
        self.resultado = {"metricas": self.metricas, "pliegues": self.tabla_pliegues, "validacion": self.predicciones_externas}
        assert len(self.predicciones_externas) == len(self.geometrias) * len(PROTOCOLOS) * (len(MODELOS) + len(self.referencias))
        assert not self.predicciones_externas.duplicated(["protocolo", "procedimiento", "id_probeta"]).any()
        self.principal = self.resumen("configuracion")
        self.transferencia = self.resumen("familia")
        self.mejor = self.principal.MAE_N.idxmin()
        self.mas_estable = self.principal.DE_MAE_pliegue_N.idxmin()
        self._terminar_etapa('evaluar_iniciales')

    def seleccionar_candidatas(self):
        """Preseleccionar las familias exclusivamente mediante validación interna."""
        if not self._iniciar_etapa('seleccionar_candidatas', 'evaluar_iniciales'):
            return

        self.planes_internos, self.cribados = {}, {}
        with threadpool_limits(limits=1):
            for protocolo, reservas in self.particiones_por_grupo.items():
                for pliegue, (entrenamiento, _) in enumerate(reservas, 1):
                    clave = (protocolo, pliegue)
                    contexto = {"protocolo": protocolo, "pliegue": pliegue}
                    self.planes_internos[clave] = self.datos_internos(entrenamiento, protocolo)
                    self.cribados[clave] = self.cribar_entrenamiento(*self.planes_internos[clave], self.configuracion, contexto, self.registros)
        self.cribado = pd.DataFrame(self.registros["cribado_interno"])
        self.seleccion = self.cribado.loc[self.cribado.seleccionada_ajuste].copy()
        self.resultado["seleccion"] = self.seleccion
        self._terminar_etapa('seleccionar_candidatas')

    def preparar_busqueda(self):
        """Fijar las combinaciones con la semilla y el presupuesto registrados."""
        if not self._iniciar_etapa('preparar_busqueda', 'seleccionar_candidatas'):
            return

        self.orden_parametros = {
            "SVR_RBF": ("C", "epsilon", "gamma"),
            "XGBoost": ("max_depth", "n_estimators", "learning_rate", "min_child_weight", "reg_lambda"),
            "CatBoost": ("iterations", "depth", "learning_rate", "l2_leaf_reg"),
            "KNN": ("n_neighbors", "weights", "p"),
            "Arbol_decision": ("max_depth", "min_samples_leaf"),
        }
        assert all(set(self.orden_parametros[f]) == set(self.configuracion["espacios"][f]) for f in self.orden_parametros)
        self.planes_busqueda = self.fijar_candidatas(self.configuracion)
        assert len(self.planes_busqueda) == 5 and all(len(candidatas) == 8 for candidatas in self.planes_busqueda.values())
        self._terminar_etapa('preparar_busqueda')

    def ajustar_hiperparametros(self):
        """Optimizar dentro de cada entrenamiento y evaluar la reserva externa."""
        if not self._iniciar_etapa('ajustar_hiperparametros', 'preparar_busqueda'):
            return

        with threadpool_limits(limits=1):
            for protocolo, reservas in self.particiones_por_grupo.items():
                for pliegue, (ind_entrenamiento, ind_prueba) in enumerate(reservas, 1):
                    clave = (protocolo, pliegue)
                    contexto_registro = {"protocolo": protocolo, "pliegue": pliegue}
                    plan = self.planes_internos[clave]
                    geometria_entrenamiento, cargas_entrenamiento = plan[:2]
                    geometria_prueba = self.geometrias.iloc[ind_prueba]
                    ganadoras, elegida, t_busqueda = self.afinar(*plan, self.configuracion, self.planes_busqueda, contexto_registro, self.registros, self.cribados[clave])
                    for puesto, receta_pliegue in enumerate(ganadoras):
                        familia = receta_pliegue["familia_real"]
                        ajuste = {**contexto_registro, "etapa_ajuste": "externa_optimizada", "id_configuracion": receta_pliegue["id_configuracion"]}
                        modelo, t_ajuste, predicciones, segundos_prediccion, prediccion_entrenamiento = self.evaluar_reserva(
                            familia, receta_pliegue["parametros"], geometria_entrenamiento, cargas_entrenamiento, geometria_prueba, ajuste, self.configuracion, self.registros)
                        inicial = self.predicciones_guardadas[protocolo, pliegue, familia]
                        procedimiento = PROCEDIMIENTOS_AJUSTADOS[puesto]
                        self._diagnosticar(modelo, familia, geometria_entrenamiento, geometria_prueba, {**contexto_registro, "procedimiento": procedimiento}, self.registros)
                        procedimientos = (procedimiento, PROCEDIMIENTOS_AJUSTADOS[2]) if puesto == elegida else (procedimiento,)
                        for proc in procedimientos:
                            self._registrar_predicciones(proc, familia, contexto_registro, self.geometrias, self.cargas, ind_entrenamiento, ind_prueba, self.nominal, predicciones, prediccion_entrenamiento,
                                    t_ajuste, segundos_prediccion, t_busqueda, inicial, self.registros)
        self._terminar_etapa('ajustar_hiperparametros')

    def reajustar_campana(self):
        """Obtener las recetas finales mediante validación interna de la campaña."""
        if not self._iniciar_etapa('reajustar_campana', 'ajustar_hiperparametros'):
            return

        ind_entrenamiento = np.arange(len(self.geometrias))
        contexto_registro = {"protocolo":"campana_completa", "pliegue":0}
        self.modelos_campana = {}
        with threadpool_limits(limits=1):
            _, _, _, internos = self.datos_internos(ind_entrenamiento, "configuracion")
            elegibles = self.cribar_entrenamiento(self.geometrias, self.cargas, self.nominal, internos, self.configuracion, contexto_registro, self.registros)
            ganadoras_ajuste, indice_elegido, _ = self.afinar(self.geometrias, self.cargas, self.nominal, internos, self.configuracion, self.planes_busqueda, contexto_registro, self.registros, elegibles)
            for w in ganadoras_ajuste:
                f = w["familia_real"]
                self.modelos_campana[f], _ = self._ajustar(
                    f, w["parametros"], self.configuracion, self.geometrias, self.cargas,
                    {**contexto_registro,"etapa_ajuste":"reajuste_campana","id_configuracion":w["id_configuracion"]}, self.registros)
        self.receta = {
            "protocolo":"configuracion", "casos_entrenamiento":self.geometrias.index.tolist(), "conjunto_variables":"A",
            "alcance":"Reajuste con todos los especímenes observados; sin evaluación externa nueva.",
            "candidatas":ganadoras_ajuste, "selector":ganadoras_ajuste[indice_elegido],
        }
        self._terminar_etapa('reajustar_campana')

    def resumir_ajuste(self):
        """Reunir métricas y contrastes emparejados tras el ajuste de hiperparámetros."""
        if not self._iniciar_etapa('resumir_ajuste', 'reajustar_campana'):
            return

        self.registros["metricas_resumen"].clear()
        self._resumir_metricas(self.registros)
        self.tablas = {nombre_tabla: pd.DataFrame(filas) for nombre_tabla, filas in self.registros.items()}
        self.resultado = {
            "metricas": self.tablas["metricas_resumen"], "validacion": self.tablas["predicciones_validacion"],
            "pliegues": self.tablas["metricas_pliegues"], "busqueda": self.tablas["busqueda_interna"],
            "cribado": self.tablas["cribado_interno"], "seleccion": self.tablas["seleccion_ajuste"],
        }
        self.metricas, self.predicciones_externas, self.tabla_pliegues = (self.resultado[k] for k in ("metricas","validacion","pliegues"))
        self.busqueda, self.seleccion, self.cribado = (
            self.resultado[k].loc[self.resultado[k].protocolo.isin(PROTOCOLOS)].copy()
            for k in ("busqueda", "seleccion", "cribado")
        )
        assert self.busqueda.groupby(["protocolo", "pliegue", "familia_real"]).size().eq(self.presupuesto).all()
        assert len(self.predicciones_externas) == len(self.geometrias) * len(PROTOCOLOS) * (len(self.modelos_iniciales) + len(self.referencias) + len(PROCEDIMIENTOS_AJUSTADOS))
        assert not self.predicciones_externas.duplicated(["protocolo", "procedimiento", "id_probeta"]).any()
        self.duracion_motor_s = perf_counter() - self.inicio_medicion

        self.mae_externo = self.metricas.pivot(index="procedimiento", columns="protocolo", values="MAE_N")
        externas_ajustadas = self.predicciones_externas.loc[self.predicciones_externas.procedimiento.isin(PROCEDIMIENTOS_AJUSTADOS)].copy()
        externas_ajustadas["inicial"] = (externas_ajustadas.predicho_inicial_misma_familia_N
                                         - externas_ajustadas.observado_N).abs()
        self.mae_pareado = (externas_ajustadas.groupby(["procedimiento", "protocolo"])
                       .agg(inicial=("inicial", "mean"), ajustada=("error_absoluto_N", "mean")))
        self._terminar_etapa('resumir_ajuste')

    def recomendar(self):
        """Aplicar la política registrada y comparar con la media por nivel."""
        if not self._iniciar_etapa('recomendar', 'resumir_ajuste'):
            return

        self.recomendacion = lectura_en_memoria(self.tablas, self.receta, self.configuracion, self.politica)

        referencia = "Media_por_nivel"
        orden_resultados = [
            referencia, "Ridge_historico", "Dummy_mediana",
            *self.mae_externo.loc[list(MODELOS)].sort_values("configuracion").index,
            "Selector_tuning",
        ]
        self.errores = self.mae_externo.loc[orden_resultados, list(PROTOCOLOS)]
        self.diferencia_referencia = self.errores.subtract(self.errores.loc[referencia], axis="columns")
        self._terminar_etapa('recomendar')

    def ajustar_ganador(self):
        """Reajustar el procedimiento de menor MAE y verificar su ecuación física."""
        if not self._iniciar_etapa('ajustar_ganador', 'recomendar'):
            return

        self.procedimiento_ganador = self.recomendacion["categorias"]["menor_MAE_procedimientos_con_ajuste"]
        if self.procedimiento_ganador in MODELOS:
            self.familia_ganadora = self.procedimiento_ganador
            parametros = self.configuracion["parametros_iniciales"][self.familia_ganadora]
            self.modelo_ganador = self.construir_modelo(self.familia_ganadora, parametros, self.configuracion).fit(self.geometrias, self.cargas)
        else:
            candidata = (self.receta["selector"] if self.procedimiento_ganador == "Selector_tuning"
                         else self.receta["candidatas"][PROCEDIMIENTOS_AJUSTADOS.index(self.procedimiento_ganador)])
            self.familia_ganadora = candidata["familia_real"]
            self.modelo_ganador = self.modelos_campana[self.familia_ganadora]

        self.ecuacion_ganadora = None
        if "base" in self.modelo_ganador.named_steps:
            intercepto, pendientes = self.ecuacion_en_unidades_originales(self.modelo_ganador)
            entradas_imputadas = self.modelo_ganador[:2].transform(self.geometrias)
            np.testing.assert_allclose(intercepto + entradas_imputadas[pendientes.index] @ pendientes,
                                       self.modelo_ganador.predict(self.geometrias), rtol=1e-10, atol=1e-7)
            self.ecuacion_ganadora = intercepto, pendientes
        self._terminar_etapa('ajustar_ganador')

    def ejemplo_mae(self):
        """Reconstruir un error individual y verificar el MAE agregado."""
        procedimientos_ejemplo = ["Media_por_nivel", "Lineal_multiple"]
        prueba_config = self.predicciones_externas.loc[
            self.predicciones_externas.protocolo.eq("configuracion") & self.predicciones_externas.procedimiento.isin(procedimientos_ejemplo),
            ["pliegue", "id_probeta", "procedimiento", "observado_N", "predicho_N"],
        ].sort_values(["pliegue", "id_probeta"]).copy()
        prueba_config["error_absoluto_N"] = (prueba_config.predicho_N - prueba_config.observado_N).abs()
        mae_config = prueba_config.groupby("procedimiento").error_absoluto_N.mean()
        np.testing.assert_allclose(mae_config.loc[procedimientos_ejemplo], self.resumen("configuracion", procedimientos_ejemplo).MAE_N)
        ejemplo = prueba_config.loc[prueba_config.id_probeta.eq(prueba_config.id_probeta.iloc[0])]
        return ejemplo, mae_config, procedimientos_ejemplo

    def ejemplo_vecinos(self):
        """Reconstruir una predicción KNN utilizando sus vecinos del entrenamiento."""
        ejemplo, _, _ = self.ejemplo_mae()
        pliegue = int(ejemplo.pliegue.iloc[0])
        caso = ejemplo.id_probeta.iloc[0]
        entrenamiento, prueba = self.particiones_por_grupo["configuracion"][pliegue - 1]
        assert caso in self.geometrias.iloc[prueba].index
        parametros_knn = self.configuracion["parametros_iniciales"]["KNN"]
        assert parametros_knn["weights"] == "uniform"
        with threadpool_limits(limits=1):
            knn_ejemplo = self.construir_modelo("KNN", parametros_knn, self.configuracion).fit(self.geometrias.iloc[entrenamiento], self.cargas.iloc[entrenamiento])
            geometria = knn_ejemplo[:-1].transform(self.geometrias.loc[[caso]])
            distancias, vecinos = knn_ejemplo.named_steps["regresor"].kneighbors(geometria)
        cargas_vecinas = self.cargas.iloc[entrenamiento].to_numpy()[vecinos[0]]
        prediccion_local = cargas_vecinas.mean()
        prediccion_registrada = self.predicciones_externas.loc[
            self.predicciones_externas.protocolo.eq("configuracion") & self.predicciones_externas.pliegue.eq(pliegue) & self.predicciones_externas.id_probeta.eq(caso) & self.predicciones_externas.procedimiento.eq("KNN"),
            "predicho_N",
        ].item()
        np.testing.assert_allclose(prediccion_local, prediccion_registrada, atol=1e-9, rtol=0)
        return distancias[0], cargas_vecinas, prediccion_local, self.cargas.loc[caso]


    def recuperar_comparacion(self, tablas):
        """Reutilizar las predicciones iniciales y escoger el menor MAE nominal."""
        predicciones = tablas["predicciones_validacion"].copy()
        metricas = tablas["metricas_resumen"].copy()
        procedimientos = (*self.modelos_iniciales, *self.referencias)
        assert set(predicciones.procedimiento) == set(procedimientos)
        assert set(metricas.protocolo) == set(PROTOCOLOS)
        assert not predicciones.duplicated(["protocolo", "procedimiento", "id_probeta"]).any()
        assert not metricas.duplicated(["protocolo", "procedimiento"]).any()
        for (protocolo, procedimiento), grupo in predicciones.groupby(["protocolo", "procedimiento"]):
            assert set(grupo.id_probeta) == set(self.cargas.index)
            np.testing.assert_allclose(grupo.observado_N, self.cargas.loc[grupo.id_probeta], atol=1e-12)
            for pliegue, reserva in grupo.groupby("pliegue"):
                prueba = self.particiones_por_grupo[protocolo][int(pliegue) - 1][1]
                assert set(reserva.id_probeta) == set(self.geometrias.iloc[prueba].index)
            calculadas = calcular_metricas(grupo.observado_N, grupo.predicho_N)
            registrada = metricas.loc[metricas.protocolo.eq(protocolo) & metricas.procedimiento.eq(procedimiento)].iloc[0]
            for campo, valor in calculadas.items():
                np.testing.assert_allclose(valor, registrada[campo], rtol=1e-12, atol=1e-12)
        self.metricas_preliminares = metricas
        ranking = metricas.loc[metricas.protocolo.eq("configuracion") & metricas.procedimiento.isin(self.modelos_iniciales)]
        self.familia_ganadora = ranking.sort_values(["MAE_N", "procedimiento"]).procedimiento.iloc[0]
        self.mejor = self.familia_ganadora
        self.metricas = metricas.loc[metricas.procedimiento.isin([self.mejor, *self.referencias])].copy()
        self.predicciones_externas = predicciones.loc[predicciones.procedimiento.isin([self.mejor, *self.referencias])].copy()
        self.predicciones_ganador = self.predicciones_externas.loc[self.predicciones_externas.procedimiento.eq(self.mejor)].copy()
        self.metricas_referencia = self.metricas.loc[self.metricas.procedimiento.isin(self.referencias)]
        self.comparacion_baseline = contraste(
            self.predicciones_externas.loc[self.predicciones_externas.protocolo.eq("configuracion")],
            self.mejor, "Media_por_nivel",
        )
        self.adopcion = adopta(self.comparacion_baseline, self.politica["banda_descriptiva_N"])
        self._etapas_completadas.add("recuperar_comparacion")


    def verificar_predicciones_seleccionadas(self, predicciones):
        """Verificar reservas y equivalencia con la evaluación inicial del ganador."""
        claves = ["protocolo", "pliegue", "id_probeta"]
        if predicciones.duplicated(claves).any():
            raise ValueError("Hay predicciones duplicadas dentro de una reserva.")
        previas = self.predicciones_ganador.set_index(claves).sort_index()
        actuales = predicciones.set_index(claves).sort_index()
        pd.testing.assert_index_equal(previas.index, actuales.index)
        np.testing.assert_allclose(
            actuales[["observado_N", "predicho_N"]],
            previas[["observado_N", "predicho_N"]], rtol=1e-10, atol=1e-8,
        )
        self.predicciones_ganador = predicciones.copy()


    def resumir_evaluacion(self, metricas_reservas, coeficientes):
        """Diagnósticos del ganador a partir de las mismas reservas verificadas."""
        claves = ["protocolo", "pliegue", "id_probeta"]
        base = self.predicciones_externas.loc[
            self.predicciones_externas.procedimiento.eq("Media_por_nivel"),
            claves + ["error_absoluto_N", "predicho_N"],
        ].rename(columns={"error_absoluto_N": "error_baseline_N", "predicho_N": "predicho_baseline_N"})
        pareadas = self.predicciones_ganador.merge(base, on=claves, validate="one_to_one")
        pareadas = pareadas.merge(
            self.geometrias[self.grupo_cols].reset_index(), on="id_probeta", validate="many_to_one"
        )
        pareadas["diferencia_error_N"] = pareadas.error_absoluto_N - pareadas.error_baseline_N
        filas = []
        for (protocolo, pliegue), grupo in pareadas.groupby(["protocolo", "pliegue"], sort=False):
            primera = grupo.iloc[0]
            etiqueta = f"Y={primera.numero_celdas_y:g}"
            if protocolo == "configuracion":
                etiqueta += f" · hf={primera.altura_fibra:g} · hp={primera.altura_pivote:g}"
            filas.append({
                "protocolo": protocolo, "pliegue": pliegue, "grupo": etiqueta,
                "MAE_modelo_N": grupo.error_absoluto_N.mean(),
                "MAE_baseline_N": grupo.error_baseline_N.mean(),
                "delta_MAE_N": grupo.diferencia_error_N.mean(),
                "RMSE_modelo_N": np.sqrt(np.mean(grupo.residuo_N ** 2)),
                "sesgo_N": -grupo.residuo_N.mean(),
            })
        grupos = pd.DataFrame(filas)
        contraste_reservas = grupos.merge(metricas_reservas, on=["protocolo", "pliegue"], validate="one_to_one")
        np.testing.assert_allclose(contraste_reservas.MAE_modelo_N, contraste_reservas.MAE_validacion_N)
        brecha = metricas_reservas.groupby("protocolo", sort=False).agg(
            entrenamiento_N=("MAE_entrenamiento_N", "mean"),
            validacion_N=("MAE_validacion_N", "mean"),
            desviacion_validacion_N=("MAE_validacion_N", "std"),
            mediana_validacion_N=("MAE_validacion_N", "median"),
        ).reset_index()
        brecha["brecha_N"] = brecha.validacion_N - brecha.entrenamiento_N
        peores = pareadas.sort_values(["protocolo", "error_absoluto_N"], ascending=[True, False])
        peores = peores.groupby("protocolo", sort=False).head(3).copy()
        sensibilidad = []
        for protocolo, grupo in pareadas.groupby("protocolo", sort=False):
            mayor = grupo.error_absoluto_N.idxmax()
            omitida = grupo.drop(index=mayor)
            comparacion = contraste(
                self.predicciones_externas.loc[self.predicciones_externas.protocolo.eq(protocolo)],
                self.mejor, "Media_por_nivel",
            )
            sensibilidad.append({
                "protocolo": protocolo,
                "MAE_completo_N": grupo.error_absoluto_N.mean(),
                "MAE_omitiendo_mayor_N": omitida.error_absoluto_N.mean(),
                "delta_sin_mayor_N": omitida.diferencia_error_N.mean(),
                "delta_completo_N": grupo.diferencia_error_N.mean(),
                "grupos_con_mejora_pct": comparacion["fraccion_grupos_mejora"] * 100,
                "delta_omitiendo_grupo_min_N": comparacion["delta_al_omitir_grupo_min_N"],
                "delta_omitiendo_grupo_max_N": comparacion["delta_al_omitir_grupo_max_N"],
            })
        resumen_coeficientes = coeficientes.groupby(["protocolo", "variable"], sort=False).agg(
            minimo=("coeficiente_N_por_unidad", "min"),
            mediana=("coeficiente_N_por_unidad", "median"),
            maximo=("coeficiente_N_por_unidad", "max"),
            reservas_retenida=("pliegue", "nunique"),
        ).reset_index()
        indice_completo = pd.MultiIndex.from_product(
            [PROTOCOLOS, self.predictores], names=["protocolo", "variable"]
        )
        resumen_coeficientes = resumen_coeficientes.set_index(["protocolo", "variable"]).reindex(indice_completo).reset_index()
        resumen_coeficientes["reservas_retenida"] = resumen_coeficientes.reservas_retenida.fillna(0)
        denominadores = metricas_reservas.groupby("protocolo").pliegue.nunique()
        resumen_coeficientes["retencion_pct"] = (
            100 * resumen_coeficientes.reservas_retenida / resumen_coeficientes.protocolo.map(denominadores)
        )
        resumen_coeficientes["unidad"] = resumen_coeficientes.variable.map(self.unidades)
        resumen_coeficientes = resumen_coeficientes.drop(columns="reservas_retenida")
        return {
            "pareadas": pareadas, "grupos": grupos, "brecha": brecha,
            "peores": peores, "sensibilidad": pd.DataFrame(sensibilidad),
            "coeficientes": resumen_coeficientes,
        }



def evaluar_configuracion_optuna(ensayo, modelo, geometrias, cargas, grupos, particiones, espacio):
    """MAE interno agrupado de una configuración, con transformaciones ajustadas por reserva."""
    from sklearn.base import clone
    parametros = {
        "regresor__fit_intercept": ensayo.suggest_categorical("fit_intercept", espacio["fit_intercept"]),
        "regresor__positive": ensayo.suggest_categorical("positive", espacio["positive"]),
    }
    if not geometrias.index.equals(cargas.index):
        raise ValueError("Geometrías y cargas deben compartir el orden de las probetas.")
    validaciones, auditoria = [], []
    with threadpool_limits(limits=1):
        for numero, (entrenamiento, validacion) in enumerate(particiones, start=1):
            if set(entrenamiento) & set(validacion) or set(grupos[entrenamiento]) & set(grupos[validacion]):
                raise ValueError("Solapamiento entre entrenamiento y validación interna.")
            if set(entrenamiento) | set(validacion) != set(range(len(geometrias))):
                raise ValueError("La partición interna debe cubrir el entrenamiento externo.")
            candidato = clone(modelo).set_params(**parametros)
            candidato.fit(geometrias.iloc[entrenamiento], cargas.iloc[entrenamiento])
            predicha = candidato.predict(geometrias.iloc[validacion])
            validaciones.append(pd.DataFrame({
                "pliegue_interno": numero,
                "id_probeta": geometrias.index[validacion],
                "observado_N": cargas.iloc[validacion].to_numpy(),
                "predicho_N": predicha,
            }))
            auditoria.append({
                "pliegue_interno": numero,
                "entrenamiento": geometrias.index[entrenamiento].tolist(),
                "validacion": geometrias.index[validacion].tolist(),
            })
    predicciones = pd.concat(validaciones, ignore_index=True)
    if predicciones.id_probeta.duplicated().any() or set(predicciones.id_probeta) != set(geometrias.index):
        raise ValueError("La evaluación interna debe predecir cada probeta de su entrenamiento externo.")
    metricas = calcular_metricas(predicciones.observado_N, predicciones.predicho_N)
    ensayo.set_user_attr("RMSE_interno_N", metricas["RMSE_N"])
    ensayo.set_user_attr("reservas_internas", auditoria)
    ensayo.set_user_attr("predicciones_internas", predicciones.to_dict(orient="records"))
    return metricas["MAE_N"]


def seleccionar_ensayo_optuna(estudio, espacio):
    """Menor MAE interno, con preferencia por la configuración inicial en empate numérico."""
    esperadas = set(product(espacio["fit_intercept"], espacio["positive"]))
    completos = [e for e in estudio.trials if e.state.name == "COMPLETE"]
    observadas = {(e.params["fit_intercept"], e.params["positive"]) for e in completos}
    if observadas != esperadas or len(completos) != len(esperadas):
        raise ValueError("La búsqueda debe completar una vez cada combinación declarada.")
    minimo = min(e.value for e in completos)
    empatados = [e for e in completos if e.value <= minimo + 1e-10]
    return min(empatados, key=lambda e: (
        not e.params["fit_intercept"], e.params["positive"], e.number,
    ))


def registrar_estudio_optuna(estudio, elegido, protocolo, pliegue, semilla):
    """Resultados de cada ensayo para las tablas y curvas de búsqueda."""
    filas, mejor = [], float("inf")
    for ensayo in sorted(estudio.trials, key=lambda e: e.number):
        mejor = min(mejor, ensayo.value)
        filas.append({
            "protocolo": protocolo, "pliegue": pliegue, "ensayo": ensayo.number + 1,
            **ensayo.params, "MAE_interno_N": ensayo.value,
            "RMSE_interno_N": ensayo.user_attrs["RMSE_interno_N"],
            "mejor_acumulado_N": mejor, "elegida": ensayo.number == elegido.number,
            "semilla_busqueda": semilla,
        })
    return filas


def comparar_resultados_optuna(predicciones_iniciales, predicciones_ajustadas, grupos, margen):
    """Comparación externa emparejada del ajuste, el modelo inicial y el baseline."""
    claves = ["protocolo", "pliegue", "id_probeta"]
    iniciales = predicciones_iniciales.loc[
        predicciones_iniciales.procedimiento.isin(["Lineal_multiple", "Media_por_nivel"])
    ].copy()
    ajustadas = predicciones_ajustadas.copy()
    ajustadas["procedimiento"] = "Lineal_optimizada"
    ajustadas["residuo_N"] = ajustadas.observado_N - ajustadas.predicho_N
    ajustadas["error_absoluto_N"] = ajustadas.residuo_N.abs()
    inicial = iniciales.loc[iniciales.procedimiento.eq("Lineal_multiple"),
                           claves + ["observado_N", "error_absoluto_N"]]
    pareadas = ajustadas.merge(inicial, on=claves, suffixes=("", "_inicial"), validate="one_to_one")
    if len(pareadas) != len(ajustadas) or len(pareadas) != len(inicial):
        raise ValueError("La comparación requiere las mismas probetas y reservas.")
    np.testing.assert_allclose(pareadas.observado_N, pareadas.observado_N_inicial, rtol=0, atol=1e-12)
    pareadas["delta_error_N"] = pareadas.error_absoluto_N - pareadas.error_absoluto_N_inicial
    comparables = pd.concat([iniciales, ajustadas], ignore_index=True)
    filas = []
    for (protocolo, procedimiento), grupo in comparables.groupby(["protocolo", "procedimiento"], sort=False):
        filas.append({"protocolo": protocolo, "procedimiento": procedimiento,
                      **calcular_metricas(grupo.observado_N, grupo.predicho_N)})
    metricas = pd.DataFrame(filas)
    por_grupo = pareadas.groupby(["protocolo", "pliegue"], sort=False).agg(
        MAE_inicial_N=("error_absoluto_N_inicial", "mean"),
        MAE_ajustado_N=("error_absoluto_N", "mean"),
        delta_MAE_N=("delta_error_N", "mean"),
    ).reset_index().merge(grupos[["protocolo", "pliegue", "grupo"]], validate="one_to_one")
    nominal = comparables.loc[comparables.protocolo.eq("configuracion")]
    frente_inicial = contraste(nominal, "Lineal_optimizada", "Lineal_multiple")
    frente_baseline = contraste(nominal, "Lineal_optimizada", "Media_por_nivel")
    return {
        "metricas": metricas, "pareadas": pareadas, "grupos": por_grupo,
        "contraste_inicial": frente_inicial, "contraste_baseline": frente_baseline,
        "adoptar_ajuste": adopta(frente_inicial, margen),
        "supera_baseline": adopta(frente_baseline, margen),
    }
