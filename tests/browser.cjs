const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const E=require('../docs/engine.js'),fixtures=JSON.parse(fs.readFileSync('docs/audit.json'));
let values=0;
function close(a,b,path=''){
 if(typeof b==='number'){assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=2e-12*Math.max(1,Math.abs(b)),`${path}: ${a} != ${b}`);values++;}
 else if(Array.isArray(b)){assert.equal(a.length,b.length,path);b.forEach((x,i)=>close(a[i],x,path+'/'+i));}
 else if(b&&typeof b==='object'){for(const k of Object.keys(b))if(k in a)close(a[k],b[k],path+'/'+k);}
 else assert.equal(a,b,path);
}
test('all Python stages and complete finite trees independently recomputed',()=>{
 for(const x of fixtures.stages)close(E.stages(x.distribution,x.config),x.trace);
 for(const x of fixtures.cases){
  const rows=E.model(x.counts);for(const s of ['start','a','b'])close(E.stages(rows[s],x.config),x.stages[s]);
  const actual=E.tree(rows,x.config,x.prompt),key=x=>JSON.stringify([x.tokens,x.stop]);
  actual.outcomes.sort((a,b)=>key(a)<key(b)?-1:1);
  const expected=structuredClone(x.sampling_tree);expected.outcomes.sort((a,b)=>key(a)<key(b)?-1:1);close(actual,expected);
  for(const b of x.beams)close(E.beam(rows,b.width,x.config,x.prompt),b.trace);
 }
 console.log(JSON.stringify({fixture_cases:fixtures.cases.length,stage_cases:fixtures.stages.length,checked_numbers:values}));
});
test('extreme temperatures, ties, domain guards and changed inputs',()=>{
 for(const temperature of [Number.MIN_VALUE,1e-300,.001,1e300]){
  const p=E.stages({a:.6,b:.4},{temperature}).output;close(Object.values(p).reduce((a,b)=>a+b,0),1);
 }
 assert.deepEqual(E.stages({b:.5,a:.5},{top_p:.5}).output,{a:1});
 for(const x of [{temperature:NaN},{temperature:0},{top_p:1.01},{top_k:1.5},{max_tokens:13}])assert.throws(()=>E.config(x));
 assert.throws(()=>E.model({start:[1,0,0],a:[0,0,0],b:[1,1,1]}));
 assert.notDeepEqual(E.stages({a:.6,b:.3,c:.1},{temperature:.5,top_p:.7}).output,E.stages({a:.6,b:.3,c:.1},{temperature:2,top_p:.7}).output);
});
