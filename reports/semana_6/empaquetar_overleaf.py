"""Empaqueta fuentes editoriales autónomos, sin documentos privados ni datos."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
ZIP=HERE/'Informe_Sprint_Semana_6_Overleaf.zip'

def main():
    files=[HERE/n for n in ['main.tex','Informe_Sprint_Semana_6.tex','referencias.bib','compilar.py']]
    files+=sorted((HERE/'secciones').glob('*.tex'))
    files+=sorted((HERE/'tablas').glob('*.tex'))
    files+=sorted((HERE/'figures').glob('*.png'))
    files+=[HERE/'figures/LOGO_FUENTE.md']
    note='''INFORME DE SPRINT — SEMANA 6
Melissa Dessire Aylas Barranca · UNI · Trabajo de Investigación II

En Overleaf: Upload Project → subir este ZIP → main.tex → pdfLaTeX.
Overleaf ejecuta BibTeX para generar las referencias IEEE.
Localmente: python compilar.py (requiere pdfLaTeX y BibTeX instalados).
No se requieren datos, código de entrenamiento ni PDF privados para compilar.

Repositorio, evidencia y PDF de referencia:
https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/sprint-semana-6/reports/semana_6

Las fuentes numéricas son los resultados históricos verificados del commit
8fdb7961d7eaaaebd9651d0b9a044168893d448a. No se reentrenaron modelos
para elaborar el informe. El README y los JSON de trazabilidad permanecen
en el repositorio; este paquete contiene los elementos para edición/compilación.
'''
    with ZipFile(ZIP,'w',ZIP_DEFLATED,compresslevel=9) as z:
        for path in files:
            z.write(path,path.relative_to(HERE).as_posix())
        z.writestr('LEEME_OVERLEAF.txt',note)
    with ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert not any('..' in Path(n).parts or n.lower().endswith('.pdf') for n in z.namelist())
    print(f'{ZIP.name}: {len(files)+1} archivos; {ZIP.stat().st_size} bytes')

if __name__=='__main__':
    main()
