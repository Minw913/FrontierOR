deliveries_out = {}
for c in cust:
    cid = str(c['id'])
    deliveries_out[cid] = {
        "demand": c['demand'],
        "total_delivered": round(cust_total_delivered[cid], 4)
    }