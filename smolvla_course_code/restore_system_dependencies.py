"""Restore system libraries when a fresh base instance reuses the project volume."""
import os
import shutil
from setup_server import Setup, SYSTEM_PACKAGES, PROXY

setup=Setup()
setup.phase='restore_system_libraries'
try:
    query=setup.run(['dpkg-query','-W','-f=${binary:Package} ${db:Status-Status}\\n'],capture=True)
    installed={line.split()[0].split(':')[0] for line in query.splitlines()
               if len(line.split())>=2 and line.split()[1]=='installed'}
    missing=[p for p in SYSTEM_PACKAGES if p not in installed]
    print('Missing system dependencies:',missing,flush=True)
    if missing:
        prefix=[] if os.geteuid()==0 else ['sudo','-n']
        if prefix and not shutil.which('sudo'):
            raise RuntimeError('System library installation requires root or configured sudo')
        setup.env['DEBIAN_FRONTEND']='noninteractive'
        apt=prefix+['apt-get']
        if PROXY:
            apt+=['-o',f'Acquire::http::Proxy={PROXY}',
                  '-o',f'Acquire::https::Proxy={PROXY}']
        setup.run(apt+['update'])
        setup.run(apt+['install','-y','--no-install-recommends',*missing],timeout=1200)
    print('SYSTEM_DEPENDENCIES_READY',flush=True)
finally:
    setup.log.close()
