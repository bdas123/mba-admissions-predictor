import json,os
start,end,step,out=53400,58900,1,"raw.jsonl"
done=set()
if os.path.exists(out):
  for l in open(out): done.add(json.loads(l)['id'])
ids=[i for i in range(start,end,step) if i not in done]
B=100
for k in range(0,len(ids),B):
  chunk=ids[k:k+B]
  urls=[f"https://gmatclub.com/forum/decision-tracker/update-{i}.html" for i in chunk]
  res=pplx_sdk.content.fetch_many(urls,concurrency=10,chunk_size=10)
  with open(out,'a') as f:
    for i,r in zip(chunk,res):
      c=''
      if r.ok and r.result and r.result.content: c=r.result.content
      # keep profile section only: from 'Similar profile' to 'Comments'
      a=c.find('Similar profile'); b=c.find('\nComments\n',a)
      seg=c[a:b if b>0 else a+8000] if a>=0 else ''
      f.write(json.dumps({'id':i,'len':len(c),'seg':seg[:8000]})+'\n')
  print(k+len(chunk),'/',len(ids),flush=True)
