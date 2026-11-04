import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {mkdtemp,rm,readdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {gunzipSync,gzipSync} from 'node:zlib';
import {createWorld} from '../dist/simulation.js';
import {validateLayout} from '../dist/layout.js';

const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('edited layouts preserve boundaries, accept posts, and stay independent of the seed',()=>{
  const world=createWorld();world.walls.push({id:'post',start:{x:450,y:450},end:{x:450,y:450},radius:25});
  assert.deepEqual(validateLayout(world),world);
  const bad=structuredClone(world);bad.walls.shift();assert.throws(()=>validateLayout(bad),/boundary/);
  const moved=structuredClone(world);moved.walls[0].start.x=20;assert.throws(()=>validateLayout(moved),/boundary/);
  const outside=structuredClone(world);outside.walls.at(-1).start.x=-10;assert.throws(()=>validateLayout(outside),/inside/);
});

test('workspace isolation, private round-trip inputs, shared example, and acknowledged cleanup', {timeout:60000}, async()=>{
  const runtime=await mkdtemp(join(tmpdir(),'collision-workspace-test-'));let child;let base;
  const a='a'.repeat(64), b='b'.repeat(64);
  async function start(){
    child=spawn(process.execPath,['dist/server.js'],{cwd:new URL('..',import.meta.url),env:{...process.env,PORT:'0',COLLISION_RUNTIME_DIR:runtime},stdio:['ignore','pipe','pipe']});
    let output='';let failure='';child.stderr.on('data',chunk=>{failure+=chunk;});
    base=await new Promise((resolve,reject)=>{
      child.stdout.on('data',chunk=>{output+=chunk;const match=output.match(/http:\/\/127\.0\.0\.1:\d+/);if(match)resolve(match[0]);});
      child.once('exit',code=>reject(Error(`Server exited ${code}: ${failure}`)));
    });
  }
  async function stop(){if(child&&child.exitCode===null){const exit=once(child,'exit');child.kill('SIGTERM');await exit;}}
  async function request(owner,path,data,method){
    return fetch(base+path,{method:method??(data===undefined?'GET':'POST'),headers:{'X-Workspace-Token':owner,'Content-Type':data instanceof Buffer?'application/gzip':'application/json'},body:data===undefined?undefined:data instanceof Buffer?data:JSON.stringify(data)});
  }
  async function result(response){assert.ok(response.ok,await response.clone().text());return response.json();}
  async function complete(id){for(let i=0;i<300;i++){const job=await result(await request(a,`/api/jobs/${id}`));if(job.status!=='running'){assert.equal(job.status,'complete',job.log);return job;}await pause(50);}throw Error('Job timeout');}
  try{
    await start();
    const state=await result(await request(a,'/api/state'));
    assert.equal(state.examples.length,3);assert.deepEqual(state.jobs,[]);
    assert.equal(state.examples.find(x=>x.kind==='dataset').details.counts.episodes,10000);
    const example=await request(b,'/api/example/example-model/model.json');
    const model=JSON.parse(gunzipSync(Buffer.from(await example.arrayBuffer())));
    assert.equal(model.training.bestEpoch,10);
    assert.equal(model.training.datasetFingerprint,state.examples.find(x=>x.kind==='balance').details.datasetFingerprint);
    assert.equal((await request(a,'/api/models/model-1793743811356-bbaf0539')).status,400);
    const world=createWorld();world.id='edited-test';world.walls.push({id:'post',start:{x:460,y:450},end:{x:460,y:450},radius:22});
    const job=await result(await request(a,'/api/collect',{episodes:30,maxActions:60,world,decisionsOnly:true,seed:42}));
    assert.equal((await request(b,`/api/jobs/${job.id}`)).status,400);
    assert.equal((await request(b,`/api/jobs/${job.id}/cancel`,{})).status,400);
    const finished=await complete(job.id);
    assert.equal(finished.result.artifact.kind,'dataset');
    assert.deepEqual(finished.result.artifact.details.world,world);
    assert.deepEqual((await result(await request(b,'/api/state'))).jobs,[]);
    // Unclaimed output survives a server restart for the owning browser.
    await stop();await start();
    assert.equal((await result(await request(a,`/api/jobs/${job.id}`))).status,'complete');
    const files={};
    for(const file of ['manifest.json','decisions.jsonl']){
      const response=await request(a,`/api/jobs/${job.id}/files/${file}`);assert.equal(response.status,200);files[file]=Buffer.from(await response.arrayBuffer());
    }
    assert.equal((await request(b,`/api/jobs/${job.id}/files/manifest.json`)).status,400);
    await result(await request(a,`/api/jobs/${job.id}/ack`,{}));
    assert.deepEqual(await readdir(runtime),[]);
    const prepared=await result(await request(a,'/api/prepare',{}));
    assert.equal((await request(b,`/api/jobs/${prepared.id}/input/manifest.json`,files['manifest.json'],'PUT')).status,400);
    for(const [file,buffer] of Object.entries(files)) await result(await request(a,`/api/jobs/${prepared.id}/input/${file}`,buffer,'PUT'));
    const balance=await result(await request(a,'/api/balance',{jobId:prepared.id,datasetId:finished.result.artifact.id,fraction:.5,seed:42}));
    const balanced=await complete(balance.id);assert.equal(balanced.result.artifact.kind,'balance');
    assert.equal(balanced.result.artifact.datasetId,finished.result.artifact.id);
    assert.equal(balanced.result.artifact.details.achievedCollisionFraction,.5);
    await result(await request(a,`/api/jobs/${balance.id}/ack`,{}));
    const invalid=createWorld();invalid.walls.shift();assert.equal((await request(a,'/api/collect',{world:invalid,episodes:10})).status,400);
    assert.deepEqual(await readdir(runtime),[]);
  }finally{await stop();await rm(runtime,{recursive:true,force:true});}
});
