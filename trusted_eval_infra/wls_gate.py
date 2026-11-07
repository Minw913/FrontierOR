"""Read-only WLS admission checks. No credentials or raw environment IDs are logged."""
import json
import time
import urllib.request

BASE = 'https://license.gurobi.com/manager/api/v1/'


def capacity(fields, *, include_environments=False):
    headers = {'Accept': 'application/json',
               'X-GUROBI-ACCESS-ID': fields['WLSACCESSID'],
               'X-GUROBI-SECRET-KEY': fields['WLSSECRET']}
    def get(endpoint):
        request = urllib.request.Request(BASE + endpoint, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    license_id = fields['LICENSEID']
    license_row = next(r for r in get('licenses') if str(r['id']) == license_id)
    now_ms = time.time() * 1000
    environments = set()
    host_environments = {}
    for row in get('sessions'):
        if (str(row.get('info', {}).get('licenseId')) == license_id
            and row.get('startedAt', 0) <= now_ms < row.get('endedAt', 0)):
            environments.add(row['containerId'])
    for row in get('tokens?length=5000'):
        if str(row.get('licenseId')) == license_id and row.get('expiresAt', 0) > now_ms:
            environments.add(row['containerId'])
            hostname = row.get('hostname')
            if hostname:
                host_environments.setdefault(hostname, set()).add(row['containerId'])
    if include_environments:
        return int(license_row['maxContainers']), environments, host_environments
    return int(license_row['maxContainers']), len(environments)


def can_admit(limit, active, local_reservations):
    # Callers may remove only confirmed one-to-one backend matches. Unknown
    # reservations remain charged to avoid an admission visibility race.
    return limit > 0 and active + local_reservations < limit


def unobserved_reservations(reservation_hosts, host_environments):
    """Deduplicate only a one-to-one hostname/environment match from WLS.

    Unknown legacy reservations, missing tokens and ambiguous identities remain
    fully charged. One backend environment can cover at most one reservation.
    """
    matched = set()
    unknown = 0
    for hostname in reservation_hosts:
        identities = host_environments.get(hostname, set()) if hostname else set()
        if len(identities) == 1 and not (identities & matched):
            matched.update(identities)
        else:
            unknown += 1
    return unknown
