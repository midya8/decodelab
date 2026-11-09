/* Independent scalar browser implementation. No Python results are used to compute outputs. */
(function(root){
'use strict';
const END='</s>',cmp=(a,b)=>a<b?-1:a>b?1:0,sum=x=>x.reduce((a,b)=>a+b,0);
function config(c={}){
 const r={max_tokens:3,temperature:1,top_k:null,top_p:null,...c};
 if(!Number.isInteger(r.max_tokens)||r.max_tokens<1||r.max_tokens>12||!Number.isFinite(r.temperature)||r.temperature<=0||
 (r.top_k!==null&&(!Number.isInteger(r.top_k)||r.top_k<1))||
 (r.top_p!==null&&(!Number.isFinite(r.top_p)||r.top_p<=0||r.top_p>1)))throw Error('Invalid decoding settings');
 return r;
}
function stages(d,c){
 c=config(c);let entries=Object.entries(d);
 if(!entries.length||entries.some(([t,p])=>!t||!Number.isFinite(p)||p<0||p>1)||Math.abs(sum(entries.map(x=>x[1]))-1)>1e-10)throw Error('Invalid distribution');
 const total=sum(entries.map(x=>x[1]));
 entries=entries.filter(x=>x[1]>0).map(([t,p])=>[t,p/total]).sort((a,b)=>b[1]-a[1]||cmp(a[0],b[0]));
 const max=Math.log(entries[0][1]);let temp=entries.map(([t,p])=>[t,Math.exp((Math.log(p)-max)/c.temperature)]);
 const z=sum(temp.map(x=>x[1]));temp=temp.map(([t,p])=>[t,p/z]);
 let selected=c.top_k===null?temp:temp.slice(0,c.top_k),kmass=sum(selected.map(x=>x[1]));
 selected=selected.map(([t,p])=>[t,p/kmass]);let kept=[];
 for(const pair of selected){if(c.top_p!==null&&c.top_p<1&&kept.length&&sum(kept.map(x=>x[1]))>=c.top_p)break;kept.push(pair);}
 const pmass=sum(kept.map(x=>x[1]));
 return {base:Object.fromEntries(entries),temperature:Object.fromEntries(temp),top_k:Object.fromEntries(selected),output:Object.fromEntries(kept.filter(x=>x[1]>0).map(([t,p])=>[t,p/pmass])),top_k_retained_temperature_mass:kmass,nucleus_retained_top_k_mass:pmass,nucleus_prefix_mass_before_last:sum(kept.slice(0,-1).map(x=>x[1]))};
}
function model(counts){
 if(!counts||Object.keys(counts).sort().join(',')!=='a,b,start')throw Error('Three rows required');
 const rows={};
 for(const [s,row] of Object.entries(counts)){
  if(!Array.isArray(row)||row.length!==3||row.some(n=>!Number.isInteger(n)||n<0||n>1000000)||!sum(row))throw Error('Use integer weights 0-1000000, positive row total');
  rows[s]=Object.fromEntries(['a','b',END].map((t,i)=>[t,row[i]/sum(row)]));
 }
 return rows;
}
function compareTokens(a,b){for(let i=0;i<Math.min(a.length,b.length);i++){const c=cmp(a[i],b[i]);if(c)return c;}return a.length-b.length;}
function tree(rows,c,prompt=[]){
 c=config(c);if(prompt.some(t=>t!=='a'&&t!=='b'))throw Error('Invalid prompt');
 const outcomes=[];let visited=0;
 function walk(tokens,mass,log,depth){if(++visited>100000)throw Error('Tree too large');
 if(depth===c.max_tokens){outcomes.push({tokens,probability:mass,model_log_probability:log,stop:'cap'});return;}
 const base=rows[tokens.length?tokens.at(-1):prompt.length?prompt.at(-1):'start'];
 for(const [t,q] of Object.entries(stages(base,c).output)){
  const nextLog=log+Math.log(base[t]);
  if(t===END){visited++;outcomes.push({tokens,probability:mass*q,model_log_probability:nextLog,stop:'eos'});}
  else walk([...tokens,t],mass*q,nextLog,depth+1);
 }}
 walk([],1,0,0);outcomes.sort((a,b)=>b.probability-a.probability||compareTokens(a.tokens,b.tokens)||cmp(a.stop,b.stop));
 return {outcomes,visited_nodes:visited,total_probability:sum(outcomes.map(x=>x.probability)),eos_probability:sum(outcomes.filter(x=>x.stop==='eos').map(x=>x.probability)),cap_probability:sum(outcomes.filter(x=>x.stop==='cap').map(x=>x.probability)),expected_generated_tokens:sum(outcomes.map(x=>x.tokens.length*x.probability))};
}
function beam(rows,width,c,prompt=[]){
 c=config(c);if(!Number.isInteger(width)||width<1||width>32)throw Error('Invalid beam width');
 let beams=[{tokens:[...prompt],log_probability:0,finished:false}],steps=[];
 for(let n=0;n<c.max_tokens;n++){
  let candidates=[];
  beams.forEach((b,parent)=>{
   if(b.finished){candidates.push({...b,parent,emitted:null});return;}
   const row=rows[b.tokens.length?b.tokens.at(-1):'start'];
   for(const [t,p] of Object.entries(row))if(p>0)candidates.push({tokens:t===END?[...b.tokens]:[...b.tokens,t],log_probability:b.log_probability+Math.log(p),finished:t===END,parent,emitted:t});
  });
  candidates.sort((a,b)=>b.log_probability-a.log_probability||compareTokens(a.tokens,b.tokens)||Number(b.finished)-Number(a.finished));
  beams=candidates.slice(0,width);steps.push({step:n+1,candidates,survivors:beams});if(beams.every(x=>x.finished))break;
 }
 return {beams,steps};
}
const api={config,stages,model,tree,beam};if(typeof module!=='undefined')module.exports=api;root.DecodeEngine=api;
})(typeof window!=='undefined'?window:globalThis);
