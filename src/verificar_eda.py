"""Verificación de la entrega EDA guardada, sin ejecutar ni editar notebooks.

Uso: ``python -m src.verificar_eda``. Devuelve código 0 si todo está conforme
y 1 si existe una incidencia. Solo escribe ``reports/verificacion_eda.json``.
No lee valores tabulares ni informa el tamaño del conjunto experimental.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
from urllib.parse import unquote, urlsplit

import nbformat
from PIL import Image


NOTEBOOKS = (
    "01_Ingesta_Curaduria_y_Calidad.ipynb",
    "02_EDA_Avanzado.ipynb",
    "03_Transformaciones_Exploratorias.ipynb",
)
REPORT_DIRS = ("eda", "eda_calidad", "eda_relaciones", "eda_transformaciones")
ANALYSIS_SOURCES = {"eda": "src/eda.py", "eda_calidad": "src/eda_calidad.py",
                    "eda_relaciones": "src/eda_relaciones.py",
                    "eda_transformaciones": "src/transformaciones_eda.py"}
INPUT_FIELDS = ("inputs", "inputs_sha256", "source", "source_sha256", "sources_sha256")
ARTIFACT_FIELDS = ("artifact_sha256", "artifacts_sha256")
HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(path, root):
    """El informe público no debe conservar rutas externas ni nombres privados."""
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "[archivo externo privado]"
    if "_private" in relative.parts or relative.suffix.lower() == ".pdf":
        return "[documento privado]"
    return relative.as_posix()


def markdown_destinations(text):
    """Extrae destinos inline y de referencia; ignora bloques/fragmentos de código.

    Admite destinos entre ángulos, paréntesis equilibrados y títulos opcionales.
    No resuelve anclas HTML ni solicita recursos de red.
    """
    text = re.sub(r"(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text)
    text = re.sub(r"(`+).*?\1", "", text, flags=re.S)
    destinations = []

    def parse_destination(fragment):
        fragment = fragment.lstrip()
        if fragment.startswith("<"):
            end = fragment.find(">")
            return fragment[1:end] if end >= 0 else ""
        chars, depth, escaped = [], 0, False
        for char in fragment:
            if escaped:
                chars.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "(" :
                depth += 1
                chars.append(char)
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
                chars.append(char)
            elif char.isspace() and depth == 0:
                break
            else:
                chars.append(char)
        return "".join(chars)

    for match in re.finditer(r"\]\(\s*", text):
        destination = parse_destination(text[match.end():])
        if destination:
            destinations.append(destination)
    for match in re.finditer(r"(?m)^\s{0,3}\[[^\]]+\]:\s*(.+)$", text):
        destination = parse_destination(match[1])
        if destination:
            destinations.append(destination)
    return list(dict.fromkeys(destinations))


def local_destination(destination, source, root):
    """Devuelve una ruta local sin fragmento/query, o None para enlaces remotos/anclas."""
    destination = destination.strip()
    if not destination or destination.startswith(("#", "//")):
        return None
    # Las rutas Windows absolutas se verifican sin copiarlas al informe público.
    if re.match(r"^[A-Za-z]:[\\/]", destination):
        return Path(unquote(destination.split("#", 1)[0])).resolve()
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    return ((Path(root) / path.lstrip("/")) if path.startswith("/")
            else Path(source).parent / path).resolve()


def hash_entries(value):
    """Normaliza un mapa ruta→SHA o un registro {path,sha256}."""
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("path"), str) and "sha256" in value:
        return [(value["path"], value["sha256"])]
    return list(value.items())


def verify(root):
    root = Path(root).resolve()
    checks = []
    notebooks, links, manifests, figures = [], [], [], []

    def check(category, path, passed, detail):
        item = {"category": category, "path": safe_name(path, root),
                "passed": bool(passed), "detail": detail}
        checks.append(item)
        return item

    for name in NOTEBOOKS:
        path = root / "notebooks" / name
        item = {"path": safe_name(path, root), "code_cells": 0, "executed_code_cells": 0}
        notebooks.append(item)
        if not path.is_file():
            check("notebook", path, False, "Falta el notebook requerido")
            continue
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            item["sha256"] = sha256_file(path)
            code = [(i + 1, c) for i, c in enumerate(notebook.cells) if c.cell_type == "code"]
            item["code_cells"] = len(code)
            item["executed_code_cells"] = sum(c.execution_count is not None for _, c in code)
            item["unexecuted_cells"] = [i for i, c in code if c.execution_count is None]
            item["error_cells"] = [i for i, c in code if any(o.output_type == "error" for o in c.get("outputs", []))]
            check("notebook_schema", path, True, "Esquema nbformat válido")
            check("notebook_execution", path, bool(code) and not item["unexecuted_cells"] and not item["error_cells"],
                  "Celdas de código ejecutadas y sin salidas de error" if code and not item["unexecuted_cells"] and not item["error_cells"]
                  else "Hay celdas sin ejecutar, salidas de error o falta código ejecutable")
        except Exception as exc:
            check("notebook_schema", path, False, f"No se pudo validar ({type(exc).__name__})")

    documents = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    for path in documents:
        if not path.is_file():
            check("markdown", path, False, "Falta documentación requerida")
            continue
        check("markdown", path, True, "Documento disponible")
        try:
            destinations = markdown_destinations(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as exc:
            check("markdown", path, False, f"Documento ilegible ({type(exc).__name__})")
            continue
        for destination in destinations:
            try:
                target = local_destination(destination, path, root)
                if target is None:
                    continue
                # Se aceptan carpetas enlazadas como índices del repositorio.
                exists = target.exists()
                item = check("local_link", target, exists,
                             "Destino local disponible" if exists else "Destino local inexistente")
                item["source_document"] = safe_name(path, root)
                links.append({"source": safe_name(path, root), "target": safe_name(target, root), "exists": exists})
            except (OSError, ValueError):
                check("local_link", path, False, "Enlace local no resoluble; destino omitido por privacidad")

    def verify_hash(path, expected, category):
        if not isinstance(expected, str) or not HASH_PATTERN.fullmatch(expected):
            return check(category, path, False, "SHA-256 ausente o inválido")
        if not path.is_file():
            return check(category, path, False, "Falta el archivo registrado en el manifiesto")
        try:
            actual = sha256_file(path)
            item = check(category, path, actual == expected.lower(),
                         "SHA-256 coincide" if actual == expected.lower() else "SHA-256 no coincide")
            item.update(expected_sha256=expected.lower(), actual_sha256=actual)
            return item
        except OSError as exc:
            return check(category, path, False, f"Archivo no legible ({type(exc).__name__})")

    for dirname in REPORT_DIRS:
        directory = root / "reports" / dirname
        manifest_path = directory / "manifest.json"
        entry = {"path": safe_name(manifest_path, root), "input_hashes_checked": 0,
                 "artifact_hashes_checked": 0, "tables_csv": len(list((directory / "tables").glob("*.csv")))}
        manifests.append(entry)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError("Manifiesto no es objeto")
            check("manifest", manifest_path, True, "JSON válido")
            entry["sha256"] = sha256_file(manifest_path)
        except (OSError, ValueError) as exc:
            check("manifest", manifest_path, False, f"Manifiesto inexistente o inválido ({type(exc).__name__})")
            continue
        if "status" in manifest:
            check("manifest_status", manifest_path, manifest["status"] == "executed", "Estado esperado: executed")
        for field in INPUT_FIELDS:
            if field not in manifest:
                continue
            values = hash_entries(manifest[field])
            if not values:
                check("input_hash", manifest_path, False, f"Campo {field} sin mapa de hashes válido")
            for relative, expected in values:
                if not isinstance(relative, str):
                    check("input_hash", manifest_path, False, "Nombre de archivo no válido")
                    continue
                entry["input_hashes_checked"] += 1
                verify_hash((root / relative).resolve(), expected, "input_hash")
        check("manifest_inputs", manifest_path, entry["input_hashes_checked"] > 0, "Manifiesto con hashes de fuentes verificables")
        for field in ARTIFACT_FIELDS:
            if field not in manifest:
                continue
            values = hash_entries(manifest[field])
            for relative, expected in values:
                if not isinstance(relative, str):
                    check("artifact_hash", manifest_path, False, "Nombre de artefacto no válido")
                    continue
                entry["artifact_hashes_checked"] += 1
                verify_hash((directory / relative).resolve(), expected, "artifact_hash")
        check("manifest_artifacts", manifest_path, entry["artifact_hashes_checked"] > 0, "Manifiesto con hashes de artefactos verificables")
        if "analysis_source_sha256" in manifest:
            analysis_source = manifest["analysis_source_sha256"]
            if isinstance(analysis_source, str):
                source_path = manifest.get("analysis_source", ANALYSIS_SOURCES[dirname])
                if isinstance(source_path, str):
                    verify_hash((root / source_path).resolve(), analysis_source, "analysis_hash")
                else:
                    check("analysis_hash_map", manifest_path, False, "Ruta de código analítico no válida")
            else:
                entries = hash_entries(analysis_source)
                check("analysis_hash_map", manifest_path, bool(entries), "Código analítico registrado con hash")
                for relative, expected in entries:
                    verify_hash((root / relative).resolve(), expected, "analysis_hash")
        elif isinstance(manifest.get("analysis_source"), dict):
            entries = hash_entries(manifest["analysis_source"])
            check("analysis_hash_map", manifest_path, bool(entries), "Código analítico registrado con hash")
            for relative, expected in entries:
                verify_hash((root / relative).resolve(), expected, "analysis_hash")

    for dirname in REPORT_DIRS:
        directory = root / "reports" / dirname / "figures"
        pngs, svgs = sorted(directory.glob("*.png")), sorted(directory.glob("*.svg"))
        check("figure_directory", directory, bool(pngs), "Figuras PNG presentes")
        for path in pngs:
            svg = path.with_suffix(".svg")
            entry = {"png": safe_name(path, root), "svg": safe_name(svg, root)}
            figures.append(entry)
            check("figure_pair", path, svg.is_file(), "SVG correspondiente disponible" if svg.is_file() else "Falta el SVG correspondiente")
            try:
                with Image.open(path) as picture:
                    dpi = picture.info.get("dpi")
                    entry["width_px"], entry["height_px"] = picture.size
                    entry["dpi"] = list(dpi) if isinstance(dpi, (tuple, list)) else dpi
                    valid_dpi = isinstance(dpi, (tuple, list)) and len(dpi) >= 2 and all(float(d) >= 299 for d in dpi[:2])
                    picture.verify()
                check("figure_png", path, True, "PNG íntegro")
                check("figure_dpi", path, valid_dpi, "Resolución declarada mínima: 299 dpi")
            except Exception as exc:
                check("figure_png", path, False, f"PNG no verificable ({type(exc).__name__})")
        for path in svgs:
            if not path.with_suffix(".png").is_file():
                check("figure_pair", path, False, "SVG sin PNG correspondiente")

    environment = {"python": platform.python_version(), "system": platform.system(),
                   "nbformat": version("nbformat"), "Pillow": version("Pillow")}
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, check=True)
        environment["git_branch"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        environment["git_branch"] = "no disponible"
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Verificación de archivos guardados; no ejecuta notebooks ni valida causalidad o desempeño predictivo",
        "environment": environment,
        "summary": {"notebooks": len(notebooks), "code_cells": sum(n["code_cells"] for n in notebooks),
                    "executed_code_cells": sum(n["executed_code_cells"] for n in notebooks),
                    "figures_png": len(figures), "tables_csv": sum(m["tables_csv"] for m in manifests),
                    "local_links": len(links), "checks": len(checks), "failed_checks": len(failures)},
        "notebooks": notebooks, "manifests": manifests, "figures": figures,
        "local_links": links, "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    result = verify(args.root)
    destination = args.root.resolve() / "reports" / "verificacion_eda.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(f"Verificación EDA: {result['status']}. Notebooks: {summary['notebooks']}; "
          f"celdas de código ejecutadas: {summary['executed_code_cells']}/{summary['code_cells']}; "
          f"figuras PNG: {summary['figures_png']}; tablas CSV: {summary['tables_csv']}.")
    for item in result["checks"]:
        if not item["passed"]:
            print(f"INCIDENCIA [{item['category']}] {item['path']}: {item['detail']}")
    print("Informe: reports/verificacion_eda.json")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
