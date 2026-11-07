"""Small explicitly allocated Docker subnets, without changing daemon settings."""
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import tempfile


def choose_subnet(used, pool='10.240.0.0/16'):
    occupied=[ipaddress.ip_network(value, strict=False) for value in used]
    for subnet in ipaddress.ip_network(pool).subnets(new_prefix=28):
        if not any(subnet.overlaps(other) for other in occupied if other.version==4):
            return str(subnet)
    raise RuntimeError('No available /28 subnet in the CORAL Docker network pool')


def create_internal_network(name, docker):
    if not name.startswith(('frontieror-agent-', 'frontieror-wls-')):
        raise ValueError('Only CORAL-owned internal networks may be allocated')
    runtime=Path(os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir()))
    fd=os.open(runtime/f'frontieror-network-allocation-{os.getuid()}.lock',os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        listed=docker(['network','ls','-q'])
        if listed.returncode:raise RuntimeError('Cannot list Docker networks')
        ids=listed.stdout.split();used=[]
        if ids:
            inspected=docker(['network','inspect',*ids])
            if inspected.returncode:raise RuntimeError('Cannot inspect Docker networks')
            for network in json.loads(inspected.stdout):
                used.extend(c['Subnet'] for c in (network.get('IPAM',{}).get('Config') or []) if c.get('Subnet'))
        routes=subprocess.run(['ip','-j','route','show'],capture_output=True,text=True,timeout=10)
        if routes.returncode:raise RuntimeError('Cannot inspect host routes before subnet allocation')
        used.extend(r['dst'] for r in json.loads(routes.stdout) if r.get('dst') not in (None,'default'))
        subnet=choose_subnet(used,os.environ.get('FRONTIER_OR_DOCKER_NETWORK_POOL','10.240.0.0/16'))
        result=docker(['network','create','--internal','--subnet',subnet,'--label','frontieror.managed=true',name])
        if result.returncode:
            raise RuntimeError(f'Docker internal network creation failed: {result.stderr.strip()}')
        return result
    finally:
        os.close(fd)
