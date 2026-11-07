"""Fuentes verificadas y preparación de los datos del benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


REVISION_DATOS = "ed6cd3385bf3e282535804c61d341e008627ab16"
URL_DATOS = (
    "https://raw.githubusercontent.com/melissaaylasb-sudo/"
    f"T02_MIA14_SeccionB/{REVISION_DATOS}/"
)
HUELLA_MANIFIESTO = "3accfa5425f81dba2e22a556dbf6b1843b658af4743c825c4b068f6b27f07ca7"
NOMBRE_CUADERNO = "Comparativa_6_Modelos_Regresion_Pantograficas.ipynb"
ARCHIVOS = {
    "muestras": "data/processed/preliminary/model_table.csv",
    "ficha_variables": "docs/benchmark/features.csv",
    "particiones": "reports/permutation_importance/tables/particiones_externas.csv",
    "historia": "docs/benchmark/antecedentes_metricas.csv",
    "previo": "reports/permutation_importance/tables/metricas_resumen.csv",
    "decisiones_fe": "reports/feature_engineering/tables/decisiones.csv",
    "experimento": "config/benchmark_regresion.json",
    "politica": "config/lectura_benchmark_regresion.json",
}
COLUMNAS_DE_ORIGEN = {
    "__source_record": "registro_origen",
    "Case_n": "id_probeta",
    "n_cells_Y": "numero_celdas_y",
    "Fiber_height": "altura_fibra",
    "Fiber_base": "ancho_fibra",
    "Fiber_total_length": "longitud_total_fibras",
    "Fiber_total_volume": "volumen_total_fibras",
    "Pivot_height": "altura_pivote",
    "Pivot_radius": "radio_pivote",
    "Pivot_total_number": "numero_total_pivotes",
    "Pivot_total_volume": "volumen_total_pivotes",
    "Sample_total_volume": "volumen_total_probeta",
    "First_Failure_Load_N": "carga_primera_falla_N",
    "Feature": "variable",
    "fold": "pliegue",
    "casos_train": "casos_entrenamiento",
    "casos_test": "casos_prueba",
    "n_train": "n_entrenamiento",
    "n_test": "n_prueba",
    "R2_OOF": "R2_validacion",
}


@dataclass
class DatosCampana:
    """Tablas y contrato metodológico cargados sin modificar sus fuentes."""

    muestras: pd.DataFrame
    ficha_variables: pd.DataFrame
    particiones: pd.DataFrame
    historia: pd.DataFrame
    previo: pd.DataFrame
    decisiones_fe: pd.DataFrame
    configuracion: dict
    politica: dict
    predictores: list[str]
    objetivo: str
    grupo_cols: list[str]
    origen: str


def _localizar_fuentes(carpeta_notebook: str | Path | None) -> tuple[Path | None, dict, str]:
    ubicacion = (
        Path(__file__).resolve().parent
        if carpeta_notebook is None else Path(carpeta_notebook).expanduser().resolve()
    )
    if not ubicacion.exists():
        raise FileNotFoundError(f"No existe la ubicación del notebook: {ubicacion}")
    cuaderno = ubicacion if ubicacion.is_file() else None
    if cuaderno is not None and cuaderno.suffix.lower() != ".ipynb":
        raise ValueError("Indica una carpeta o un archivo .ipynb.")
    carpeta = cuaderno.parent if cuaderno is not None else ubicacion
    raiz = next(
        (ruta for ruta in (carpeta, *carpeta.parents)
         if (ruta / ARCHIVOS["experimento"]).is_file()),
        None,
    )
    if raiz is not None:
        return raiz, {}, "checkout local"
    if cuaderno is None:
        original = carpeta / NOMBRE_CUADERNO
        candidatos = [original] if original.is_file() else list(carpeta.glob("*.ipynb"))
        cuaderno = candidatos[0] if len(candidatos) == 1 else None
    copia = {}
    if cuaderno is not None:
        contenido = json.loads(cuaderno.read_text(encoding="utf-8-sig"))
        copia = contenido.get("metadata", {}).get("fuentes_benchmark", {})
    if copia:
        if copia.get("revision") != REVISION_DATOS:
            raise ValueError("La revisión de la copia adjunta al notebook difiere del protocolo.")
        return None, copia, "copia del notebook"
    return None, {}, "GitHub (revisión fijada)"


def _leer_entradas(raiz: Path | None, copia: dict, origen: str) -> dict[str, str]:
    entradas = {}
    for nombre, ruta in ARCHIVOS.items():
        try:
            if raiz is not None:
                contenido = (raiz / ruta).read_bytes()
            elif copia:
                entradas[nombre] = copia["archivos_utf8"][ruta]
                continue
            else:
                with urlopen(URL_DATOS + ruta, timeout=20) as respuesta:
                    contenido = respuesta.read()
            entradas[nombre] = contenido.decode("utf-8-sig").replace("\r\n", "\n")
        except (OSError, KeyError) as exc:
            raise RuntimeError(
                f"No se pudo leer {ruta} desde {origen}. Usa el notebook publicado "
                "completo, su checkout o una conexión a GitHub."
            ) from exc
    manifiesto = "\n".join(sorted(
        f"{ARCHIVOS[nombre]} {hashlib.sha256(texto.encode()).hexdigest()}"
        for nombre, texto in entradas.items()
    ))
    if hashlib.sha256(manifiesto.encode()).hexdigest() != HUELLA_MANIFIESTO:
        raise ValueError(
            f"Algún insumo difiere de la revisión {REVISION_DATOS[:7]}. "
            "Revisa los cambios locales antes de comparar métricas."
        )
    return entradas


def _validar_datos(datos: DatosCampana) -> None:
    """Mantiene las comprobaciones de objetivo, predictores y reservas originales."""
    muestras = datos.muestras
    predictores = datos.predictores
    if datos.objetivo != "carga_primera_falla_N" or datos.configuracion["unidad"] != "N":
        raise ValueError("La variable objetivo debe ser la carga de primera falla en N.")
    if len(predictores) != len(set(predictores)) or not set(predictores).issubset(muestras):
        raise ValueError("Los predictores deben ser únicos y existir en la tabla de muestras.")
    excluidas = {datos.objetivo, "id_probeta", "registro_origen", "carga_terminal", "desplazamiento_maximo"}
    if set(predictores).intersection(excluidas):
        raise ValueError("Los predictores contienen identificadores o variables de respuesta.")
    if not muestras.id_probeta.is_unique or not muestras[datos.objetivo].notna().all():
        raise ValueError("Las probetas deben ser únicas y tener carga observada.")
    if not set(datos.grupo_cols).issubset(muestras):
        raise ValueError("Faltan columnas que definen la configuración nominal.")
    casos = set(muestras.id_probeta)
    for _, fila in datos.particiones.iterrows():
        entrenamiento, prueba = (
            set(map(int, str(fila[columna]).split(";")))
            for columna in ("casos_entrenamiento", "casos_prueba")
        )
        if entrenamiento & prueba or entrenamiento | prueba != casos:
            raise ValueError("Una partición solapa entrenamiento y prueba o no cubre las probetas.")


def cargar_datos(carpeta_notebook: str | Path | None = None) -> DatosCampana:
    """Carga fuentes verificadas desde la carpeta del módulo o una ruta explícita.

    Acepta la carpeta del notebook o su archivo .ipynb. Prioriza un checkout
    completo, la copia incluida en los metadatos y la revisión fija de GitHub.
    Conserva el parseo de precisión completa del cuaderno original.
    """
    raiz, copia, origen = _localizar_fuentes(carpeta_notebook)
    entradas = _leer_entradas(raiz, copia, origen)
    tablas = {
        nombre: pd.read_csv(StringIO(entradas[nombre]), float_precision="round_trip")
        .rename(columns=COLUMNAS_DE_ORIGEN)
        for nombre in ("muestras", "ficha_variables", "particiones", "historia", "previo", "decisiones_fe")
    }
    ficha = tablas["ficha_variables"]
    ficha["variable"] = ficha["variable"].map(COLUMNAS_DE_ORIGEN.__getitem__)
    configuracion = json.loads(entradas["experimento"])
    objetivo = COLUMNAS_DE_ORIGEN[configuracion.pop("target")]
    configuracion["variable_objetivo"] = objetivo
    datos = DatosCampana(
        **tablas,
        configuracion=configuracion,
        politica=json.loads(entradas["politica"]),
        predictores=ficha["variable"].tolist(),
        objetivo=objetivo,
        grupo_cols=["numero_celdas_y", "altura_fibra", "altura_pivote"],
        origen=origen,
    )
    _validar_datos(datos)
    return datos


HUELLA_RESULTADOS_PREVIOS = "a3cccbc39420f019d49243d1184f9f2f7af47ca5b373c1ea5ebc8e498bd08558"


def cargar_resultados_previos(carpeta_notebook=None):
    """Recupera las predicciones y métricas verificadas de la comparación inicial."""
    ubicacion = Path(carpeta_notebook or Path(__file__).resolve().parent)
    cuaderno = ubicacion if ubicacion.is_file() else ubicacion / NOMBRE_CUADERNO
    if not cuaderno.is_file() and ubicacion.is_dir():
        candidatos = list(ubicacion.glob("*.ipynb"))
        if len(candidatos) == 1:
            cuaderno = candidatos[0]
    metadata = json.loads(cuaderno.read_text(encoding="utf-8-sig"))["metadata"]
    copia = metadata["resultados_preliminares"]
    huella = hashlib.sha256(copia["contenido"].encode()).hexdigest()
    if huella != HUELLA_RESULTADOS_PREVIOS or copia["sha256"] != huella:
        raise ValueError("Las predicciones previas difieren de la versión verificada.")
    return {clave: pd.DataFrame(filas) for clave, filas in json.loads(copia["contenido"]).items()}
