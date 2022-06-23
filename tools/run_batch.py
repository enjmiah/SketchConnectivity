#!/usr/bin/env python

"""
Detect the intended connectivity of some line drawings.
"""

import argparse
import ast
import asyncio
from contextlib import contextmanager
from datetime import datetime
import locale
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import jinja2
import yaml

import _sketching as _s
sys.path.insert(0, os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/../python/'))

# Check for pdflatex.
if not shutil.which('pdflatex'):
    print('Could not find pdflatex (is it in your PATH?)', file=sys.stderr)
    sys.exit(1)


ROW_PY = os.path.abspath(os.path.split(os.path.abspath(__file__))[0]+'/run_pipeline.py')

lock = threading.Lock()
processes = []

TEMPLATE = R'''
\pdfsuppresswarningpagegroup=1

\documentclass[12pt,letterpaper]{article}
\usepackage[paperheight=32in, paperwidth=15in, margin=2mm]{geometry}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{tikz}
\setlength{\parindent}{0em}
\setlength{\parskip}{0em}

\begin{document}

\pagenumbering{gobble}

\begin{centering}

{% for row in rows %}
  {\Large \lstinline!{{ row['id'] }}!} \\
  \bigskip

  \begin{tabular}{c|c|c}

    \textbf{Input} &
    \textbf{Trivial closed loops} &
    \textbf{Our result} \\

    {{ overlay_figures(row['consolidated_centerlines']) }} &
    {{ overlay_figures(row['v0_regions'], row['consolidated_centerlines']) }} &
    {{ overlay_figures(row['v7_regions'], row['consolidated_centerlines'], row['v7_cc_conn'], row['v7_annotations_conn_all']) }} \\
  \end{tabular}

  \vspace{1em}
{% endfor %}

\end{centering}

\end{document}
'''

def fix_path(path: str) -> str:
    """Convert backslashes to forward slashes, to satisfy LaTeX."""
    """And convert to absolute path so we can still find the file after cwd in subprocess."""
    return os.path.abspath(path).replace('\\', '/')

@contextmanager
def temporary_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        # Avoid returning paths like "C:\Users\ADMIN-~1" which break pdflatex.
        yield os.path.realpath(tmpdir)
    except:
        print('\n*\n* An error occurred. The intermediate files are left in ' +
              f'\n* "{tmpdir}".\n*\n', file=sys.stderr)
        raise
    # Unlike tempfile.TemporaryDirectory, we do not want to remove the temporary
    # directory when an exception occurs because we want to be able to manually
    # inspect the directory afterwards to see what went wrong.
    shutil.rmtree(tmpdir)

def tex_graphics(path: str) -> str:
    if not path:
        return '\\small(missing)'
    return fR'\includegraphics[height=7.5in, width=3.5in, keepaspectratio]{{{fix_path(path)}}}'


# We wrap single figures in a TikZ as well to get the exact same margin /
# positioning.
def overlay_figures(*paths) -> str:
    for path in paths:
        if not path:
            return '\\small(missing)'
    builder = [R'\begin{tikzpicture} ']
    for path in paths:
        builder.append(fR'\node at (0,0) {{{tex_graphics(path)}}};')
    builder.append(R'\end{tikzpicture}')
    return ''.join(builder)


### Asyncio
async def read_stream(stream):
    lines = []
    while not stream.at_eof():
        data = await stream.readline()
        line = data.decode(locale.getpreferredencoding(False))
        if len(line):
            lines.append(line)
    return lines


async def run_subprocess_in_bg(case, cmd, timeout):
    # print("Running {} `{}`".format(case, " ".join([str(c) for c in cmd])))
    print("Running {}".format(case))
    # https://stackoverflow.com/questions/45769985/asyncio-create-subprocess-exec-console-window-opening-for-each-call
    # This flag avoids popup window after crashing
    DETACHED_PROCESS = 0x00000008
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        creationflags=DETACHED_PROCESS,
    )

    with lock:
        processes.append(process)

    task_code = asyncio.ensure_future(process.wait())
    task_out = asyncio.ensure_future(read_stream(process.stdout))
    task_err = asyncio.ensure_future(read_stream(process.stderr))

    done, pending = await asyncio.wait([task_code, task_out, task_err], timeout=timeout)
    if pending:
        # timeout
        if process.returncode is None:
            # kill the subprocess, then `await future` will return soon
            try:
                print(f'* Killing {case} after {timeout} s...')
                process.kill()
            except ProcessLookupError:
                pass

    code = await task_code
    out = await task_out
    err = await task_err

    if code == 0:
        print(f'* Success: {case}.')
    else:
        print(f'* Failure: {case} {code}...')

    return code, out, err, case


async def run_test(case, cmd, timeout, semaphore):
    # prevent more than certain number things to run at the same time
    async with semaphore:
        return await run_subprocess_in_bg(case, cmd, timeout)


async def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='YAML file listing drawings')
    parser.add_argument('-o', '--output',
                        help='name to use for the output summary PDF')
    parser.add_argument('-s', '--snapshot', default='snapshot',
                        help='output directory for results (default: "snapshot")')
    parser.add_argument('--timeout', type=int, default=240,
                        help='timeout for processing each drawing in seconds (default: 240)')
    parser.add_argument('--max-procs', type=int, default=5,
                        help='maximum number of drawings to process in parallel (default: 5)')
    args = parser.parse_args()

    with open(args.input) as f:
        input_ = yaml.safe_load(f)

    os.makedirs(os.path.join(args.snapshot, "plane"), exist_ok=True)
    os.makedirs(os.path.join(args.snapshot, "candidates"), exist_ok=True)
    os.makedirs(os.path.join(args.snapshot, "v4"), exist_ok=True)
    os.makedirs(os.path.join(args.snapshot, "v5"), exist_ok=True)
    os.makedirs(os.path.join(args.snapshot, "v7"), exist_ok=True)

    rows = []
    semaphore = asyncio.Semaphore(args.max_procs)

    with temporary_directory() as tmpdir:
        coroutines = []

        root = input_.get('root', os.path.dirname(args.input))
        # Create tasks
        for path in input_['drawings']:
            if isinstance(path, dict):
                path = path['path']

            cmd = [sys.executable, ROW_PY, path, '--root', root,
                   '--tmpdir', tmpdir]
            if args.snapshot:
                cmd += ['--snapshot', args.snapshot]
            coroutines.append(run_test(path, cmd, args.timeout, semaphore=semaphore))

        # Run
        results = await asyncio.gather(*coroutines)

        # Read out
        for code, out, err, case in results:
            print('======================================')

            overall_out = err + out
            entry_line = [l for l in out if 'Entry: ' in l and 'Partial Entry: ' not in l]
            if code == 0 and len(entry_line) > 0:
                entry_line = entry_line[0].replace('Entry: ', '')
                entry = ast.literal_eval(entry_line)
                rows.append(entry)

                if len(overall_out) >= 3:
                    print(f'{case} ran in {overall_out[-2]}')
                else:
                    print(f'{case} ran.')
            else:
                print(f'{case} failed...')
                print('{}'.format(''.join(overall_out[-10::])))
                print('...')

                entry_line = [l for l in err if 'Partial Entry: ' in l]
                if len(entry_line) > 0:
                    entry_line = entry_line[-1].replace('Partial Entry: ', '')
                    entry = ast.literal_eval(entry_line)
                    rows.append(entry)

                    if len(overall_out) >= 3:
                        print(f'{case} crashed in {overall_out[-3]}')
                    else:
                        print(f'{case} crashed.')

        fname = 'compilation'
        template = jinja2.Template(TEMPLATE)

        with open(os.path.join(tmpdir, f'{fname}.tex'), 'w') as f:
            f.write(template.render(rows=rows, tex_graphics=tex_graphics,
                                    overlay_figures=overlay_figures))
        subprocess.check_call(['pdflatex', f'{fname}.tex'],
                              cwd=tmpdir, stdout=subprocess.DEVNULL)
        date = datetime.today().strftime('%Y-%m-%d')
        output_path = args.output if args.output else f'{fname}-{date}.pdf'
        shutil.move(os.path.join(tmpdir, f'{fname}.pdf'), output_path)


if __name__ == '__main__':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.get_event_loop().run_until_complete(main())
    except KeyboardInterrupt:
        with lock:
            for proc in processes:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
