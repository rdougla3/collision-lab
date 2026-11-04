import { distanceToSegment } from './geometry.js';
import { BOUNDARIES, getLayout, setLayout, validateLayout } from './layout.js';
import { createWorld, Simulation } from './simulation.js';
import type { Vec2, World, Wall } from './types.js';

type Drag = { kind:'move'|'start'|'end'|'radius'|'goal'; origin:Vec2; wall?:Wall; original:World; goal?:Vec2 };
export class ArenaEditor {
  private selected: string | null = null;
  private tool: 'select'|'wall'|'post' = 'select';
  private drag?: Drag;
  private preview?: World;
  private goalEnabled = false;
  constructor(private canvas: HTMLCanvasElement, private callbacks: {
    pause:()=>void; preview:(world:World)=>void; draw:()=>void; goal:()=>Vec2|undefined; moveGoal:(point:Vec2)=>void;
  }) {
    for(const [id,tool] of [['edit-select','select'],['add-wall','wall'],['add-post','post']] as const) {
      document.getElementById(id)!.addEventListener('click',()=>{this.tool=tool;this.selected=null;this.controls();this.callbacks.draw();});
    }
    document.getElementById('delete-obstacle')!.addEventListener('click',()=>this.remove());
    document.getElementById('reset-layout')!.addEventListener('click',()=>{
      this.commit(createWorld());
      (document.querySelector('.layout-menu') as HTMLDetailsElement).open=false;
    });
    canvas.addEventListener('pointerdown',event=>this.down(event));
    canvas.addEventListener('pointermove',event=>this.move(event));
    canvas.addEventListener('pointerup',()=>this.up());
    canvas.addEventListener('pointercancel',()=>this.cancel());
    canvas.addEventListener('keydown',event=>{
      if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();this.remove();}
      if(event.key==='Escape'){this.cancel();this.selected=null;this.tool='select';this.controls();this.callbacks.draw();}
    });
    document.addEventListener('lab:layout',()=>{
      if(!getLayout().walls.some(w=>w.id===this.selected))this.selected=null;
      this.controls();this.callbacks.draw();
    });
    this.controls();
  }
  setGoalEnabled(enabled:boolean):void {this.goalEnabled=enabled;if(!enabled){if(this.selected==='goal')this.selected=null;}this.controls();}
  private controls():void {
    for(const [id,tool] of [['edit-select','select'],['add-wall','wall'],['add-post','post']]) {
      const button=document.getElementById(id!)!;button.classList.toggle('primary',this.tool===tool);button.setAttribute('aria-pressed',String(this.tool===tool));
    }
    (document.getElementById('delete-obstacle') as HTMLButtonElement).disabled=!this.selected || this.selected==='goal';
    const hint=document.getElementById('editor-hint')!;
    hint.hidden=this.tool==='select';
    hint.textContent=this.tool==='wall'?'Tap to place a wall · Esc to cancel':'Tap to place a round obstacle · Esc to cancel';
  }
  private point(event:PointerEvent):Vec2 {
    const bounds=this.canvas.getBoundingClientRect();const world=getLayout();
    return {x:(event.clientX-bounds.left)/bounds.width*world.width,y:(1-(event.clientY-bounds.top)/bounds.height)*world.height};
  }
  private tolerance():number {return 12*getLayout().width/this.canvas.getBoundingClientRect().width;}
  private near(a:Vec2,b:Vec2):boolean {return Math.hypot(a.x-b.x,a.y-b.y)<=this.tolerance();}
  private down(event:PointerEvent):void {
    if(event.button!==0)return;
    this.canvas.focus({preventScroll:true});const point=this.point(event);const world=getLayout();
    if(this.tool==='wall'||this.tool==='post') {
      if(world.walls.length>=44){this.error(new Error('The arena supports up to 40 obstacles.'));return;}
      const post=this.tool==='post';
      const wall:Wall={id:`obstacle-${crypto.randomUUID()}`,start:{x:point.x-(post?0:90),y:point.y},end:{x:point.x+(post?0:90),y:point.y},radius:post?28:8};
      world.walls.push(wall);
      if(this.commit(world)){this.selected=wall.id;this.tool='select';this.controls();this.callbacks.draw();}return;
    }
    const goal=this.callbacks.goal();
    // Goal takes precedence over obstacle handles and has no resize/delete handles.
    if(goal&&this.near(point,goal)){
      this.callbacks.pause();this.selected='goal';this.drag={kind:'goal',origin:point,original:world,goal:{...goal}};
    }else{
      let wall=world.walls.find(w=>w.id===this.selected);
      let kind:Drag['kind']='move';
      if(wall){
        const post=wall.start.x===wall.end.x&&wall.start.y===wall.end.y;
        if(post&&this.near(point,{x:wall.start.x+wall.radius,y:wall.start.y}))kind='radius';
        else if(!post&&this.near(point,wall.start))kind='start';
        else if(!post&&this.near(point,wall.end))kind='end';
        else wall=undefined;
      }
      wall??=[...world.walls].reverse().find(w=>!BOUNDARIES.includes(w.id)&&distanceToSegment(point,w.start,w.end)<=w.radius+this.tolerance());
      if(!wall){this.selected=null;this.controls();this.callbacks.draw();return;}
      this.callbacks.pause();this.selected=wall.id;this.drag={kind,origin:point,wall:structuredClone(wall),original:world};
    }
    this.canvas.setPointerCapture(event.pointerId);this.controls();this.callbacks.draw();
  }
  private move(event:PointerEvent):void {
    if(!this.drag)return;
    const point=this.point(event);const drag=this.drag;
    if(drag.kind==='goal'){
      try{this.callbacks.moveGoal(point);this.callbacks.draw();}catch{/* Keep the last valid goal position. */}return;
    }
    const world=structuredClone(drag.original);const wall=world.walls.find(w=>w.id===drag.wall!.id)!;
    if(drag.kind==='move'){
      const dx=point.x-drag.origin.x,dy=point.y-drag.origin.y;
      wall.start={x:drag.wall!.start.x+dx,y:drag.wall!.start.y+dy};wall.end={x:drag.wall!.end.x+dx,y:drag.wall!.end.y+dy};
    }else if(drag.kind==='radius')wall.radius=Math.max(5,Math.min(80,Math.hypot(point.x-wall.start.x,point.y-wall.start.y)));
    else wall[drag.kind]=point;
    this.preview=world;this.callbacks.preview(world);
  }
  private up():void {
    if(!this.drag)return;
    const preview=this.preview;this.drag=undefined;this.preview=undefined;
    if(preview)this.commit(preview);else this.callbacks.draw();
  }
  private cancel():void {this.drag=undefined;this.preview=undefined;this.callbacks.preview(getLayout());}
  private commit(world:World):boolean {
    try{
      const validated=validateLayout(world);const goal=this.callbacks.goal();
      if(goal&&!new Simulation({},validated).isFree(goal,10))throw new Error('Move the goal before placing an obstacle over it.');
      validated.id=`edited-${crypto.randomUUID()}`;
      this.callbacks.pause();setLayout(validated);this.error('');return true;
    }catch(error){this.callbacks.preview(getLayout());this.error(error);return false;}
  }
  private remove():void {
    if(!this.selected||this.selected==='goal'||BOUNDARIES.includes(this.selected))return;
    const world=getLayout();world.walls=world.walls.filter(w=>w.id!==this.selected);this.commit(world);this.selected=null;this.controls();this.callbacks.draw();
  }
  private error(reason:unknown):void {document.getElementById('error')!.textContent=reason instanceof Error?reason.message:String(reason);}
  draw(context:CanvasRenderingContext2D):void {
    if(!this.selected||this.selected==='goal')return;
    const wall=(this.preview??getLayout()).walls.find(w=>w.id===this.selected);if(!wall)return;
    context.strokeStyle='#ffc878';context.lineWidth=wall.radius*2+5;context.lineCap='round';
    context.beginPath();context.moveTo(wall.start.x,wall.start.y);context.lineTo(wall.end.x,wall.end.y);context.stroke();
    if(wall.start.x===wall.end.x&&wall.start.y===wall.end.y){context.beginPath();context.arc(wall.start.x,wall.start.y,wall.radius+2,0,Math.PI*2);context.lineWidth=3;context.stroke();}
    const points=wall.start.x===wall.end.x&&wall.start.y===wall.end.y?[{x:wall.start.x+wall.radius,y:wall.start.y}]:[wall.start,wall.end];
    for(const p of points){context.fillStyle='#ffc878';context.beginPath();context.arc(p.x,p.y,Math.max(7,this.tolerance()*.4),0,Math.PI*2);context.fill();}
  }
}
