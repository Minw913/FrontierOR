"""Bounded public-file tools and native OpenRouter usage evidence for EOH."""
from pathlib import Path
import datetime,fcntl,json,os,stat,time,uuid
import requests

MAX_CHUNK=8192
MAX_TOOL_ROUNDS=16
TOOL={'type':'function','function':{'name':'read_public_file','description':'Read a bounded byte range from a listed public task/tiny/development file. No host paths, final tests, checkers or reference files are available.','parameters':{'type':'object','properties':{'path':{'type':'string'},'offset':{'type':'integer','minimum':0},'length':{'type':'integer','minimum':1,'maximum':MAX_CHUNK}},'required':['path'],'additionalProperties':False}}}

def append_json(path,row):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row)+'\n');f.flush()
    p.chmod(0o600)

def read_public_file(public_root,arguments):
    if not isinstance(arguments, dict):raise ValueError('Tool arguments must be an object')
    root=Path(public_root).resolve();manifest=json.loads((root/'public_manifest.json').read_text())
    name=arguments.get('path');offset=arguments.get('offset',0);length=arguments.get('length',MAX_CHUNK)
    if not isinstance(name, str) or name not in manifest['files'] or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('File is not in the public manifest')
    if type(offset) is not int or offset<0 or type(length) is not int or not 1<=length<=MAX_CHUNK:
        raise ValueError('Invalid byte range')
    target=root/name
    if target.is_symlink() or not target.resolve().is_relative_to(root):raise ValueError('Invalid public file path')
    fd=os.open(target,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):raise ValueError('Not a regular public file')
        data=os.pread(fd,length,offset)
    finally:os.close(fd)
    return {'path':name,'offset':offset,'next_offset':offset+len(data),'total_bytes':info.st_size,'eof':offset+len(data)>=info.st_size,'text':data.decode('utf-8',errors='replace')}

def generate(messages,*,model,key,endpoint,output_dir,public_root=None,post=None,
             max_tokens=32768,timeout=600,max_tool_rounds=MAX_TOOL_ROUNDS,usage_path=None):
    """Record every response, including charged tool-call responses, before parsing."""
    post=post or requests.post
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    abort=out/'credit-abort.json'
    if abort.exists():raise RuntimeError('EOH halted after an OpenRouter credit error')
    call_id=uuid.uuid4().hex;trace=out/'llm_calls'/call_id;trace.mkdir(parents=True,mode=0o700)
    messages=list(messages)
    read_count=0
    for turn in range(max_tool_rounds+1):
        payload={'model':model,'messages':messages,'temperature':0.8,'max_tokens':max_tokens}
        if public_root and turn<max_tool_rounds:payload['tools']=[TOOL]
        (trace/f'{turn:02d}-request.json').write_text(json.dumps(payload,indent=2)+'\n')
        response=None
        for attempt in range(3):
            if abort.exists():raise RuntimeError('EOH halted after an OpenRouter credit error')
            try:
                response=post(endpoint,json=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'frontier-or/1.0'},timeout=timeout)
                status=response.status_code
                if status==402:
                    abort.write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':402,'call_id':call_id})+'\n')
                    raise RuntimeError('OpenRouter credit error (402); EOH halted')
                response.raise_for_status();data=response.json()
                if data.get('error'):raise RuntimeError('OpenRouter response contains an API error')
                break
            except (requests.RequestException,ValueError) as exc:
                append_json(out/'api_errors.jsonl',{'call_id':call_id,'turn':turn,'attempt':attempt,'error_type':type(exc).__name__,'status':getattr(response,'status_code',None),'billing_unknown':isinstance(exc,requests.RequestException) and response is None})
                if attempt==2:raise
                if response is not None and 400<=response.status_code<500 and response.status_code!=429:raise
                time.sleep(5)
        append_json(usage_path or out/'api_usage.jsonl',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response_id':data.get('id'),'model':data.get('model',model),'usage':data.get('usage') or {},'call_id':call_id,'tool_turn':turn})
        message=data['choices'][0]['message']
        (trace/f'{turn:02d}-response.json').write_text(json.dumps(data,indent=2)+'\n')
        calls=message.get('tool_calls') or []
        if not calls:
            content=message.get('content')
            if not content:raise RuntimeError('Model returned no program text; reasoning alone is not a program')
            return content
        if not public_root or turn>=max_tool_rounds:raise RuntimeError('Public-file tool budget exhausted')
        messages.append(message)
        for call in calls:
            try:
                if call['function']['name']!='read_public_file':raise ValueError('Unknown tool')
                read_count += 1
                if read_count > 32:raise ValueError('Public file read budget exhausted')
                args=json.loads(call['function']['arguments']);result=read_public_file(public_root,args)
                append_json(out/'public_reads.jsonl',{'call_id':call_id,'response_id':data.get('id'),'path':result['path'],'offset':result['offset'],'next_offset':result['next_offset']})
            except (ValueError,TypeError,KeyError,OSError) as exc:result={'error':str(exc)}
            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
    raise RuntimeError('Public-file tool budget exhausted')
