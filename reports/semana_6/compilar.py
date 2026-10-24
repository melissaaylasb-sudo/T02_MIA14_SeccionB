"""Compila el informe con pdfLaTeX y BibTeX sin entrenar ni consultar datos."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex-bin',type=Path,help='Directorio de pdflatex y bibtex si no están en PATH')
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    job='Informe_Sprint_Semana_6'
    build=here/'.build'
    build.mkdir(exist_ok=True)
    for name in [job+'.tex','referencias.bib']:
        shutil.copy2(here/name,build/name)
    for name in ['secciones','figures','tablas']:
        shutil.copytree(here/name,build/name,dirs_exist_ok=True)
    env=os.environ.copy()
    if args.tex_bin:
        env['PATH']=str(args.tex_bin)+os.pathsep+env.get('PATH','')
    exe={name:shutil.which(name,path=env['PATH']) for name in ('pdflatex','bibtex')}
    if not all(exe.values()):
        raise SystemExit('Instale TeX Live/MiKTeX o indique --tex-bin. En Overleaf use main.tex.')
    commands=[
        [exe['pdflatex'],'-interaction=nonstopmode','-halt-on-error','-file-line-error',job+'.tex'],
        [exe['bibtex'],job],
        [exe['pdflatex'],'-interaction=nonstopmode','-halt-on-error','-file-line-error',job+'.tex'],
        [exe['pdflatex'],'-interaction=nonstopmode','-halt-on-error','-file-line-error',job+'.tex'],
    ]
    for i,cmd in enumerate(commands,1):
        process=subprocess.run(cmd,cwd=build,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        output=here/f'compilacion_{i}.txt'
        output.write_bytes(process.stdout)
        if process.returncode:
            raise SystemExit(f'Fallo en la pasada {i}; consulte {output.name}.')
    log=(build/(job+'.log')).read_text(encoding='utf-8',errors='replace')
    problems=[t for t in ('Overfull \\hbox','Overfull \\vbox','undefined references','undefined on input','Rerun to get') if t in log]
    if problems:
        raise SystemExit('Revisar diagnósticos: '+', '.join(problems))
    # Evita escribir sobre el PDF mientras un visor lo está leyendo.
    # Reemplazo atómico tras completar todas las pasadas; se conserva el previo
    # si Windows mantiene un bloqueo que impide incluso renombrar el archivo.
    for attempt in range(4):
        try:
            os.replace(build/(job+'.pdf'),here/(job+'.pdf'))
            break
        except PermissionError:
            if attempt==3:
                raise SystemExit('PDF compilado en .build; cierre el visor y vuelva a ejecutar para publicar el archivo final.')
            time.sleep(0.5)
    for suffix in ['.aux','.bbl','.blg','.log','.out','.toc']:
        source=build/(job+suffix)
        if source.exists():
            shutil.copy2(source,here/source.name)
    print(f'Compilación completa: {here/(job+".pdf")}')

if __name__=='__main__':
    main()
