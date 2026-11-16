const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function harness() {
  class E {
    constructor(){this.value='';this.textContent='';this.listeners={};this.files=[];}
    addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
    dispatchEvent(event){event.target=this;for(const fn of this.listeners[event.type]||[])fn(event);}
  }
  const ids=['input','conversation','call-id','call-date','example','upload-call','message','download-output','json'];
  const roots=Object.fromEntries(ids.map(x=>[x,new E()]));
  const call={call_id:'editor-only',started_at:'2026-11-15T12:00:00+03:00',turns:[{idx:7,speaker:'client',text:'  интырнет  не робит '},{idx:41,speaker:'operator',text:'Проверяю.'}]};
  roots.input.value=JSON.stringify(call);let changes=0;roots.input.addEventListener('input',()=>changes++);
  const sandbox={document:{getElementById:id=>roots[id]},window:{},Event:class{constructor(type){this.type=type;}},Date,Blob,URL};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../frontend/simple.js'),'utf8'),sandbox);
  return {roots,call,api:sandbox.window.compactFactEditor,changes:()=>changes};
}
test('editing metadata preserves imported sparse turn indices and noisy text exactly',()=>{
  const h=harness();h.roots['call-id'].value='changed';h.roots['call-id'].dispatchEvent({type:'input'});
  const value=JSON.parse(h.roots.input.value);assert.deepEqual(value.turns,h.call.turns);assert.equal(value.started_at,h.call.started_at);assert.equal(h.changes(),1);
});
test('manual conversation parses roles and multiline ASR without spell correction',()=>{
  const h=harness();const out=JSON.parse(JSON.stringify(h.api.parseTurns('Клиент: не робит, эээ\nи вчера тоже\nОператор: Уточню.')));
  assert.deepEqual(out,[{idx:0,speaker:'client',text:'не робит, эээ\nи вчера тоже'},{idx:1,speaker:'operator',text:'Уточню.'}]);
});
test('invalid human edit invalidates the previous successful request',()=>{
  const h=harness();h.roots.conversation.value='нет роли';h.roots.conversation.dispatchEvent({type:'input'});
  assert.equal(h.roots.input.value,'');assert.equal(h.changes(),1);assert.match(h.roots.message.textContent,/роль/);
});
test('upload discards gold facts and preserves original metadata and turns',async()=>{
  const h=harness();const source={...h.call,facts:[{answer:'never send gold'}]};
  h.roots['upload-call'].files=[{size:200,text:async()=>JSON.stringify(source)}];
  await h.roots['upload-call'].listeners.change[0]({target:h.roots['upload-call']});
  assert.deepEqual(JSON.parse(h.roots.input.value),h.call);assert.match(h.roots.message.textContent,/исключены/);
});
