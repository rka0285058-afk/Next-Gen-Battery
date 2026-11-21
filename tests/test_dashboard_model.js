const fs=require('fs');
const html=fs.readFileSync(require('path').join(__dirname,'..','dashboard','index.html'),'utf8');
const m=html.match(/<script id="model">([\s\S]*?)<\/script>/)[1];
const MODEL=new Function(m+';return MODEL;')();
function fresh(){const s={w:Object.assign({},MODEL.WEIGHTS['Long-range EV']),region:'Global',shock:0,prog:1,gst:false,mult:{},on:{},I:Object.assign({},MODEL.INDIA_DEF)};MODEL.CH.forEach(p=>{s.mult[p.name]={cost:1,cyc:1};s.on[p.name]=true});return s;}
function show(tag,S){const t=Date.now();const R=MODEL.compute(S);console.log('==',tag,(Date.now()-t)+'ms');R.rows.forEach(r=>console.log(r.short.padEnd(14),'p1',r.p1.toFixed(3),'p3',r.p3.toFixed(3),'score',r.score.mean.toFixed(3),'c35',r.cost.p50.toFixed(0),'wh',r.wh.toFixed(0),'cyc',r.cyc.toFixed(0),'lcos',r.lcos.toFixed(2),'ratio',r.ratio.toFixed(3)));}
for(const [uc,reg] of [['Long-range EV','Global'],['Grid storage','Global'],['Low-cost mobility','India'],['Grid storage','India']]){const S=fresh();S.w=Object.assign({},MODEL.WEIGHTS[uc]);S.region=reg;show(uc+' '+reg,S);}
let S=fresh();S.shock=150;show('Li +150% (EV)',S);
S=fresh();S.mult['Na-ion']={cost:1.25,cyc:.35};S.w=Object.assign({},MODEL.WEIGHTS['Grid storage']);show('Na falls short (grid)',S);
S=fresh();MODEL.CH.forEach(p=>S.on[p.name]=false);S.on['LFP']=true;S.on['Na-ion']=true;show('two chems',S);
