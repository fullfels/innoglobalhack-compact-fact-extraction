const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness() {
  class Element {
    constructor() {
      this.children=[]; this.value=''; this.textContent=''; this.disabled=false;
      const classes=new Set();
      this.classList={add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x)};
    }
    replaceChildren(...nodes) { this.children=nodes; this.textContent=''; }
    append(node) { this.children.push(node); }
    scrollIntoView() { this.scrolled=true; }
  }
  const roots=Object.fromEntries(['input','example','facts','turns','timeline','json','message','run','status','model-banner'].map(id=>[id,new Element()]));
  roots.example.value='0';
  function nodes() { const found=[]; function visit(n){found.push(n);n.children.forEach(visit);} Object.values(roots).forEach(visit);return found; }
  const pending=[];
  const sandbox={document:{getElementById:id=>roots[id]||nodes().find(n=>n.id===id),createElement:()=>new Element(),querySelectorAll:()=>nodes().filter(n=>n.classList.contains('highlight'))},
    performance:{now:()=>10},setInterval:()=>{},fetch:(url,opts)=>{
      if(url==='/health')return Promise.resolve({json:async()=>({ready:true,is_mock:true,model_state:'ready'})});
      return new Promise((resolve,reject)=>pending.push({resolve,reject,payload:JSON.parse(opts.body)}));
    }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../frontend/app.js'),'utf8'),sandbox);
  return {roots,pending};
}
function result(call, overrides={}) {
  const t=call.turns[0];
  return {meta:{is_mock:true,warnings:[]},prediction:{call_id:call.call_id,turns:call.turns,
    facts:[{object:{name:'Object'},attribute:'service_state',value:'Value',polarity:1,speaker:t.speaker,valid_from:call.started_at,valid_to:null,source:{turn_idx:t.idx,quote:t.text}}],...overrides}};
}
async function finish(h,request,body) { request.resolve({ok:true,json:async()=>body}); }

test('changing example clears facts, source, timeline and JSON',async()=>{
  const h=harness(); const work=h.roots.run.onclick(); const req=h.pending[0];
  await finish(h,req,result(req.payload));await work;
  assert.equal(h.roots.facts.children.length,1);
  assert.match(h.roots.json.textContent,/demo-001/);
  h.roots.example.value='1';h.roots.example.onchange();
  for(const id of ['facts','turns','timeline','json']) {
    assert.equal(h.roots[id].children.length,0);assert.equal(h.roots[id].textContent,'');
  }
  assert.match(h.roots.input.value,/demo-002/);
  assert.match(h.roots.message.textContent,/Ввод изменён/);
});

test('editing during fetch suppresses stale response, and new request renders normally',async()=>{
  const h=harness();const old=h.roots.run.onclick();const req=h.pending[0];
  h.roots.example.value='1';h.roots.example.onchange();
  assert.equal(h.roots.run.disabled,true);
  await finish(h,req,result(req.payload));await old;
  assert.equal(h.roots.facts.children.length,0);assert.equal(h.roots.json.textContent,'');
  assert.equal(h.roots.run.disabled,false);
  const next=h.roots.run.onclick();const second=h.pending[1];
  await finish(h,second,result(second.payload));await next;
  assert.equal(h.roots.facts.children.length,1);assert.match(h.roots.json.textContent,/demo-002/);
  h.roots.facts.children[0].onclick();
  assert.equal(h.roots.turns.children[0].classList.contains('highlight'),true);
  assert.equal(h.roots.turns.children[0].scrolled,true);
});

test('manual edit invalidates input even when reverted; stale network error is suppressed',async()=>{
  const h=harness();const saved=h.roots.input.value;
  const work=h.roots.run.onclick();
  h.roots.input.value='{}';h.roots.input.oninput();
  h.roots.input.value=saved;h.roots.input.oninput();
  h.pending[0].reject(new Error('old network failure'));await work;
  assert.equal(h.roots.json.textContent,'');assert.equal(h.roots.facts.children.length,0);
  assert.match(h.roots.message.textContent,/Ввод изменён/);
  assert.doesNotMatch(h.roots.message.textContent,/old network failure/);
});

test('different response call_id is never presented as current facts',async()=>{
  const h=harness();const work=h.roots.run.onclick();const req=h.pending[0];
  await finish(h,req,result(req.payload,{call_id:'wrong-call'}));await work;
  assert.equal(h.roots.facts.children.length,0);assert.equal(h.roots.json.textContent,'');
  assert.match(h.roots.message.textContent,/другому call_id/);
});

test('empty valid response keeps JSON viewer available',async()=>{
  const h=harness();const work=h.roots.run.onclick();const req=h.pending[0];
  await finish(h,req,result(req.payload,{facts:[]}));await work;
  assert.equal(h.roots.facts.children.length,0);assert.match(h.roots.json.textContent,/demo-001/);
  assert.equal(h.roots.run.disabled,false);
});
