import{D as ct,A as ue,aZ as S,j as it,eh as C,ew as D,ej as _,gE as K,gs as z,eB as fe,gp as O,aD as he,f0 as de,gt as pe,P as ge,f1 as me,eE as G,eo as nt,i as ut,eq as P,eZ as ft,fy as R,Q as ht,fo as dt,aq as pt,as as gt,at as mt,au as It,fG as wt,aC as xt,fI as kt,fJ as bt,aK as yt,aR as Et,b0 as St,b5 as Rt,eC as Ie,b6 as we,e$ as vt,b_ as xe,aY as ke,eD as be,bh as ye,Y as Tt,ac as Ee,eI as Se,aO as Re,hH as ve,hI as Te,hJ as Ne,hK as Me,hL as Pe,dn as Ce,r as Fe,cl as Ve,bv as Nt,gq as X,bB as Mt,eR as De,eS as Ae,hM as Oe,bH as qe,bI as Le,eQ as $e,hN as ze,hO as Ge,hP as We,hQ as je,hR as _e,hS as Be,hT as Ke,hU as Ue,hV as st,hW as Ze,hX as He,hY as Xe,bR as Pt,bT as Ct,ga as Ft,ev as H,hZ as Qe,bV as Vt,h_ as L}from"./index-Dq_Lnxi8.js";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A(n,e){Array.isArray(n)||(n=[n]),n.forEach(t=>{t!=null&&ct(t.dtype!=="complex64",()=>`${e} does not support complex64 tensors in the CPU backend.`)})}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dt(n){const e=new Float32Array(n.length);for(let t=0;t<n.length;++t)e[t]=Math.abs(n[t]);return e}const Ye=n=>{const{x:e}=n.inputs,t=n.backend;A(e,"abs");let s=new Float32Array(S(e.shape));const o=t.data.get(e.dataId).values;return s=Dt(o),t.makeOutput(s,e.shape,e.dtype)},cs={kernelName:ue,backendName:"cpu",kernelFunc:Ye};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function v(n){return(e,t,s,o,l)=>{const r=it(e,t),c=r.length,a=C(r),i=S(r),f=D(l,i),u=e.length,g=t.length,m=C(e),d=C(t),w=_(e,r),h=_(t,r);if(w.length+h.length===0)for(let p=0;p<f.length;++p)f[p]=n(s[p%s.length],o[p%o.length]);else for(let p=0;p<f.length;++p){const I=K(p,c,a),k=I.slice(-u);w.forEach(E=>k[E]=0);const x=z(k,u,m),b=I.slice(-g);h.forEach(E=>b[E]=0);const y=z(b,g,d);f[p]=n(s[x],o[y])}return[f,r]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U(n){const{inputs:e,backend:t}=n,{real:s,imag:o}=e,l=t.data.get(s.dataId).values,r=t.data.get(o.dataId).values,c=t.makeTensorInfo(s.shape,"complex64"),a=t.data.get(c.dataId);return a.complexTensorInfos={real:t.makeTensorInfo(s.shape,"float32",l),imag:t.makeTensorInfo(o.shape,"float32",r)},c}const is={kernelName:fe,backendName:"cpu",kernelFunc:U};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Q(n,e,t="float32"){if(t==="complex64"){const o=Q(n,e,"float32"),l=Q(n,e,"float32");return U({inputs:{real:o,imag:l},backend:n})}const s=O(S(e),t);return n.makeTensorInfo(e,t,s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Y(n){const{inputs:e,backend:t}=n,{x:s}=e;return t.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const us={kernelName:he,backendName:"cpu",kernelFunc:Y};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function At(n){const{inputs:e,backend:t}=n,{input:s}=e,o=t.data.get(s.dataId).complexTensorInfos.real,l=t.data.get(o.dataId).values;return t.makeTensorInfo(o.shape,o.dtype,l)}const fs={kernelName:de,backendName:"cpu",kernelFunc:At};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ot(n,e,t,s){if(s==="int32"){const o=Int32Array.from(n);return[e,"int32",o]}if(s==="bool"){const o=pe([0],t),[l,r]=v((c,a)=>c!==a?1:0)(e,[],n,o,"bool");return[r,"bool",l]}throw new Error(`Error in Cast: failed to cast ${t} to ${s}`)}function W(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{dtype:l}=s;if(l==="complex64"){if(o.dtype==="complex64")return Y({inputs:{x:o},backend:t});const f=Q(t,o.shape,o.dtype),u=W({inputs:{x:o},backend:t,attrs:{dtype:"float32"}}),g=U({inputs:{real:u,imag:f},backend:t});return t.disposeIntermediateTensorInfo(f),t.disposeIntermediateTensorInfo(u),g}if(o.dtype==="complex64"){const f=At({inputs:{input:o},backend:t}),u=W({inputs:{x:f},backend:t,attrs:{dtype:l}});return t.disposeIntermediateTensorInfo(f),u}if(!me(o.dtype,l)){const f=Y({inputs:{x:o},backend:t});return{dataId:f.dataId,shape:f.shape,dtype:l}}const r=t.data.get(o.dataId).values,[c,a,i]=Ot(r,o.shape,o.dtype,l);return t.makeTensorInfo(c,a,i)}const hs={kernelName:ge,backendName:"cpu",kernelFunc:W};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function T(n,e,t,s){return t==null?({inputs:o,backend:l})=>{const{a:r,b:c}=o,a=l;A([r,c],n);const i=a.data.get(r.dataId).values,f=a.data.get(c.dataId).values,u=r.dtype==="string"?G(i):i,g=r.dtype==="string"?G(f):f,m=s||r.dtype,[d,w]=e(r.shape,c.shape,u,g,m);return a.makeTensorInfo(w,m,d)}:({inputs:o,backend:l})=>{const{a:r,b:c}=o,a=l;if(r.dtype==="complex64"||c.dtype==="complex64"){const i=W({inputs:{x:r},backend:a,attrs:{dtype:"complex64"}}),f=a.data.get(i.dataId),u=f.complexTensorInfos.real,g=f.complexTensorInfos.imag,m=a.data.get(u.dataId).values,d=a.data.get(g.dataId).values,w=W({inputs:{x:c},backend:a,attrs:{dtype:"complex64"}}),h=a.data.get(w.dataId),p=h.complexTensorInfos.real,I=h.complexTensorInfos.imag,k=a.data.get(p.dataId).values,x=a.data.get(I.dataId).values,[b,y,E]=t(r.shape,c.shape,m,d,k,x),N=a.makeTensorInfo(E,"float32",b),q=a.makeTensorInfo(E,"float32",y),j=U({inputs:{real:N,imag:q},backend:a});return a.disposeIntermediateTensorInfo(i),a.disposeIntermediateTensorInfo(w),a.disposeIntermediateTensorInfo(N),a.disposeIntermediateTensorInfo(q),j}else{const i=a.data.get(r.dataId).values,f=a.data.get(c.dataId).values,u=s||r.dtype,[g,m]=e(r.shape,c.shape,i,f,u);return a.makeTensorInfo(m,u,g)}}}function J(n){return(e,t,s,o,l,r)=>{const c=it(e,t),a=S(c),i=c.length,f=C(c),u=D("float32",a),g=D("float32",a),m=_(e,c),d=_(t,c),w=nt(s,o),h=nt(l,r),p=e.length,I=C(e),k=t.length,x=C(t);if(m.length+d.length===0)for(let b=0;b<u.length;b++){const y=b%w.length,E=b%h.length,N=n(w[y*2],w[y*2+1],h[E*2],h[E*2+1]);u[b]=N.real,g[b]=N.imag}else for(let b=0;b<u.length;b++){const y=K(b,i,f),E=y.slice(-p);m.forEach(Z=>E[Z]=0);const N=z(E,p,I),q=y.slice(-k);d.forEach(Z=>q[Z]=0);const j=z(q,k,x),et=n(w[N*2],w[N*2+1],h[j*2],h[j*2+1]);u[b]=et.real,g[b]=et.imag}return[u,g,c]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qt=v(((n,e)=>n+e)),Je=J(((n,e,t,s)=>({real:n+t,imag:e+s}))),tn=T(ut,qt,Je),ds={kernelName:ut,backendName:"cpu",kernelFunc:tn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function en(n,e,t,s,o){const l=S(s),r=O(o,t);for(let c=0;c<n.length;c++){const a=n[c];if(a<0)throw new Error("Input x must be non-negative!");a>=o||(l>0?r[a]+=e[c]:r[a]+=1)}return r}function nn(n,e,t,s=!1){const o=n.shape[0],l=n.shape[1],r=P([o,t],e.dtype);for(let c=0;c<o;c++)for(let a=0;a<l;a++){const i=n.get(c,a);if(i<0)throw new Error("Input x must be non-negative!");i>=t||(s?r.set(1,c,i):e.size>0?r.set(r.get(c,i)+e.get(c,a),c,i):r.set(r.get(c,i)+1,c,i))}return r}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lt=v(((n,e)=>n&e)),sn=T(ft,Lt),ps={kernelName:ft,backendName:"cpu",kernelFunc:sn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F(n){return(e,t,s)=>{const o=R(t,e.length);for(let l=0;l<e.length;++l)o[l]=n(e[l],s);return o}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $t(n,e,t){const s=F(e);return V(n,s,t)}function V(n,e,t){return({inputs:s,attrs:o,backend:l})=>{const{x:r}=s;A(r,n);const c=l,a=c.data.get(r.dataId).values;let i;if(r.dtype==="string"){if(!Array.isArray(a))throw new Error("String tensor's value was not an instance of Array");i=G(a)}else i=a;const f=t||r.dtype,u=e(i,f,o);return c.makeTensorInfo(r.shape,f,u)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zt=F(n=>Math.ceil(n)),on=V(ht,zt),gs={kernelName:ht,backendName:"cpu",kernelFunc:on};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function an(n,e,t,s){const o=R(t,S(e));if(s&&t!=="string"){let l=0;n.forEach(r=>{const c=S(r.shape);o.set(r.vals,l),l+=c})}else{let l=0;n.forEach(r=>{const c=t==="string"?G(r.vals):r.vals;let a=0;for(let i=0;i<r.shape[0];++i){const f=i*e[1]+l;for(let u=0;u<r.shape[1];++u)o[f+u]=c[a++]}l+=r.shape[1]})}return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gt=v((n,e)=>n===e?1:0),rn=T(dt,Gt,null,"bool"),ms={kernelName:dt,backendName:"cpu",kernelFunc:rn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wt=F(n=>Math.exp(n)),ln=V(pt,Wt,"float32"),Is={kernelName:pt,backendName:"cpu",kernelFunc:ln};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jt=F(n=>Math.expm1(n)),cn=V(gt,jt),ws={kernelName:gt,backendName:"cpu",kernelFunc:cn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _t=F(n=>Math.floor(n)),un=V(mt,_t),xs={kernelName:mt,backendName:"cpu",kernelFunc:un};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bt=v((n,e)=>Math.floor(n/e)),fn=T(It,Bt,null,"int32"),ks={kernelName:It,backendName:"cpu",kernelFunc:fn};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hn(n,e,t,s,o,l,r,c,a){const i=P([s,l],t);for(let f=0;f<s;f++){const u=[];let g=0;for(let m=0;m<o;m++){const d=n[f*o+m];g+=d*r[m],u.push(d)}if(g<0||g>=a/l)throw new Error(`Invalid indices: ${u} does not index into ${c}`);for(let m=0;m<l;m++)i.values[f*l+m]=e.get(...e.indexToLoc(g*l+m))}return i}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dn(n,e,t){const s=P(t,n.dtype);for(let o=0;o<s.size;++o){const r=s.indexToLoc(o).slice(),c=r[0],a=r[2],i=e.locToIndex([c,a]);r[2]=e.values[i];const f=n.locToIndex(r);0<=f&&f<n.values.length&&(s.values[o]=n.values[f])}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kt=v((n,e)=>n>e?1:0),pn=T(wt,Kt,null,"bool"),bs={kernelName:wt,backendName:"cpu",kernelFunc:pn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ut=v((n,e)=>n>=e?1:0),gn=T(xt,Ut,null,"bool"),ys={kernelName:xt,backendName:"cpu",kernelFunc:gn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zt=v((n,e)=>n<e?1:0),mn=T(kt,Zt,null,"bool"),Es={kernelName:kt,backendName:"cpu",kernelFunc:mn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ht=v((n,e)=>n<=e?1:0),In=T(bt,Ht,null,"bool"),Ss={kernelName:bt,backendName:"cpu",kernelFunc:In};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wn(n,e,t){const s=(e-n)/(t-1),o=O(t,"float32");o[0]=n;for(let l=1;l<o.length;l++)o[l]=o[l-1]+s;return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xt=F(n=>Math.log(n)),xn=V(yt,Xt),Rs={kernelName:yt,backendName:"cpu",kernelFunc:xn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kn(n,e,t,s){const o=D(s,S(t));for(let l=0;l<o.length;++l){const r=l*e;let c=n[r];for(let a=0;a<e;++a){const i=n[r+a];(Number.isNaN(i)||i>c)&&(c=i)}o[l]=c}return o}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qt=v(((n,e)=>Math.max(n,e))),bn=T(Et,Qt),vs={kernelName:Et,backendName:"cpu",kernelFunc:bn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yt=v(((n,e)=>Math.min(n,e))),yn=T(St,Yt),Ts={kernelName:St,backendName:"cpu",kernelFunc:yn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tt=v(((n,e)=>n*e)),En=J(((n,e,t,s)=>({real:n*t-e*s,imag:n*s+e*t}))),Sn=T(Rt,tt,En),Ns={kernelName:Rt,backendName:"cpu",kernelFunc:Sn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jt(n,e,t){const s=Ie(-1,t);return tt([],e,s,n,t)}function Rn(n){const{inputs:e,backend:t}=n,{x:s}=e;A(s,"neg");const o=t.data.get(s.dataId).values,[l,r]=Jt(o,s.shape,s.dtype);return t.makeTensorInfo(r,s.dtype,l)}const Ms={kernelName:we,backendName:"cpu",kernelFunc:Rn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const te=v(((n,e)=>n!==e?1:0)),vn=T(vt,te,null,"bool"),Ps={kernelName:vt,backendName:"cpu",kernelFunc:vn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ee(n,e,t,s,o){const l=e.length,r=S(e),c=C(e),a=C(o),i=D(t,S(o));for(let f=0;f<r;++f){const u=K(f,l,c),g=new Array(u.length);for(let d=0;d<g.length;d++)g[d]=u[s[d]];const m=z(g,l,a);i[m]=n[f]}return i}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ne(n){const{inputs:e,attrs:t,backend:s}=n,{x:o}=e,{perm:l}=t;A(o,"transpose");const r=o.shape.length,c=new Array(r);for(let u=0;u<c.length;u++)c[u]=o.shape[l[u]];const a=s.data.get(o.dataId).values,i=ee(a,o.shape,o.dtype,l,c);return{dataId:s.write(i,c,o.dtype),shape:c,dtype:o.dtype}}const Cs={kernelName:xe,backendName:"cpu",kernelFunc:ne};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function se(n,e,t,s){const[o,l]=ke(n,s),r=be(e,"int32"),c=O(S(o),r),a=S(l);for(let i=0;i<c.length;++i){const f=i*a;let u=1;for(let g=0;g<a;++g)u*=t[f+g];c[i]=u}return{outVals:c,outShape:o,outDtype:r}}function Tn(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{axis:l,keepDims:r}=s;A(o,"prod");const c=o.shape.length,a=Tt(l,o.shape),i=Ee(a,c);let f=a,u=o;const g=[];i!=null&&(u=ne({inputs:{x:o},backend:t,attrs:{perm:i}}),g.push(u),f=Se(f.length,c));const m=t.data.get(u.dataId).values,{outVals:d,outShape:w,outDtype:h}=se(u.shape,u.dtype,m,f);let p=w;return r&&(p=Re(w,a)),g.forEach(I=>t.disposeIntermediateTensorInfo(I)),t.makeTensorInfo(p,h,d)}const Fs={kernelName:ye,backendName:"cpu",kernelFunc:Tn};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nn(n,e,t){n.forEach((s,o)=>{if(s<0||s>=t){const l=K(o,e.length,C(e)).join(",");throw new Error(`indices[${l}] = ${s} is not in [0, ${t})`)}})}function Mn(n,e){for(let t=0;t<n.length;++t){const s=n[t],o=t===n.length-1?e:n[t+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>o)throw new Error("Ragged splits must not point past values");for(let l=1;l<s.length;++l)if(s[l-1]>s[l])throw new Error("Ragged splits must be sorted in ascending order")}}function Pn(n,e,t,s){const o=[];let l=0;const r=e.length-1+t.length,c=new Array(r).fill(null).map(()=>[0]);Mn(t,s);let a=1;for(let i=0;i<e.length-1;++i){a*=e[i];const f=e[i+1];for(let u=1;u<a+1;++u)c[i].push(u*f)}for(let i=0;i<n.length;++i){let f=n[i],u=n[i]+1;for(let g=0;g<t.length;++g){const m=t[g],d=g+e.length-1;if(d>=0){const w=c[d],h=w[w.length-1]-m[f];for(let p=f;p<u;++p)c[d].push(m[p+1]+h)}f=m[f],u=m[u]}u!==f&&(o.push([f,u]),l+=u-f)}return{outSplits:c,valueSlices:o,numValues:l}}function Cn(n){const e=[];for(let t=0;t<n.length;++t){const s=n[t].length,o=R("int32",s);e.push(o),n[t].forEach((l,r)=>o[r]=l)}return e}function ot(n,e){const t=n.slice(0,e);for(;t.length<e;)t.push(1);for(let s=e;s<n.length;s++)t[e-1]*=n[s];return t}function Fn(n,e,t,s,o,l){const r=ot(e,2)[1],c=ot(l,2)[1];let a=0;for(const i of t)for(let f=i[0];f<i[1];++f){for(let u=0;u<s;++u)o[a*c+u]=n[f*r+u];++a}}function Vn(n,e,t,s,o){const l=e.slice();l[0]=o;const r=R(t,S(l)),c=n.length,a=c===0?0:c/e[0];return Fn(n,e,s,a,r,l),[r,l]}function Dn(n,e,t,s,o,l,r,c){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const a=e[0][0]-1;if(Nn(l,r,a),s.length===0)throw new Error("params.rank must be nonzero");const i=s[0],{outSplits:f,valueSlices:u,numValues:g}=Pn(l,r,n,i),m=Cn(f),d=Vn(t,s,o,u,g);return[m,d[0],d[1]]}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const at=2147483647;function An(n,e,t,s,o,l,r){if(e.length>1)throw new Error("starts must be a scalar or vector");if(o.length>1)throw new Error("limits must be a scalar or vector");if(r.length>1)throw new Error("deltas must be a scalar or vector");const c=e.length===0,a=o.length===0,i=r.length===0,f=[];c||f.push(e[0]),a||f.push(o[0]),i||f.push(r[0]);for(let h=1;h<f.length;++h)if(f[h]!==f[h-1])throw new Error("starts, limits, and deltas must have the same shape");const u=f.length===0?1:f[0],g=R("int32",u+1);g[0]=0;for(let h=0;h<u;++h){const p=c?n[0]:n[h],I=a?s[0]:s[h],k=i?l[0]:l[h];if(k===0)throw new Error("Requires delta != 0");let x;if(k>0&&I<p||k<0&&I>p)x=0;else if(x=Math.ceil(Math.abs((I-p)/k)),x>at)throw new Error(`Requires ((limit - start) / delta) <= ${at}`);g[h+1]=g[h]+x}const m=g[u],d=R(t,m);let w=0;for(let h=0;h<u;++h){const p=g[h+1]-g[h];let I=c?n[0]:n[h];const k=i?l[0]:l[h];for(let x=0;x<p;++x)d[w++]=I,I+=k}return[g,d]}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var M=Ne;class B{constructor(e,t,s,o,l,r,c,a,i,f){this.shape=e,this.shapeShape=t,this.values=s,this.valuesShape=o,this.valuesDType=l,this.defaultValue=r,this.defaultValueShape=c,this.rowPartitionValues=a,this.rowPartitionValuesShapes=i,this.rowPartitionTypes=ve(f),this.raggedRank=Te(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===M.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===M.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case M.VALUE_ROWIDS:return B.getMaxWidthValueRowID(t);case M.ROW_SPLITS:return B.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${M[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let s=0;for(let o=0;o<t-1;++o){const l=e[o+1]-e[o];l>s&&(s=l)}return s}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let s=0,o=e[0],l=0;for(let r=1;r<t;++r){const c=e[r];c!==o&&(o=c,l=Math.max(r-s,l),s=r)}return Math.max(t-s,l)}tensorShapeFromTensor(e,t,s=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return lt(e,s)}calculateOutputSize(e){const t=this.valuesShape,s=this.defaultValueShape;Me(s,t);const o=this.tensorShapeFromTensor(this.shape,this.shapeShape),r=Pe(this.raggedRank,o,t);r[0]<0&&(r[0]=e);for(let c=1;c<=this.raggedRank;++c)r[c]<0&&(r[c]=this.getMaxWidth(c));return r}calculateFirstParentOutputIndex(e,t,s){const o=Math.min(e,s),l=[];let r=0;for(let c=0;c<o;++c,r+=t)l.push(r);for(let c=o;c<e;++c)l.push(-1);return ct(l.length===e,()=>"Final length of result must be equal to firstDimension."),l}calculateOutputIndexRowSplit(e,t,s,o){const l=e.length,r=[];for(let c=0;c<l-1;++c){const a=e[c+1]-e[c];let i=Math.min(o,a),f=t[c];f===-1&&(i=0);for(let u=0;u<i;++u)r.push(f),f+=s;for(let u=0;u<a-i;++u)r.push(-1)}if(l>0&&r.length!==e[l-1])throw new Error("Invalid row split size.");return r}calculateOutputIndexValueRowID(e,t,s,o){const l=e.length,r=[];if(l===0)return[];let c=0,a=e[0];if(a>=t.length)throw new Error(`Got currentValueRowId=${a}, which is not less than ${t.length}`);let i=t[a];r.push(i);for(let f=1;f<l;++f){const u=e[f];if(u===a)i>=0&&(++c,c<o?i+=s:i=-1);else{if(c=0,a=u,u>=t.length)throw new Error(`Got nextValueRowId=${u} which is not less than ${t.length}`);i=t[u]}r.push(i)}if(r.length!==e.length)throw new Error("Invalid row ids.");return r}calculateOutputIndex(e,t,s,o){const l=this.getRowPartitionTensor(e),r=this.getRowPartitionTypeByDimension(e);switch(r){case M.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(l,t,s,o);case M.ROW_SPLITS:if(l.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${l.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(l,t,s,o);default:throw new Error(`Unsupported partition type: ${M[r]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case M.FIRST_DIM_SIZE:return e[0];case M.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case M.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${M[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),s=this.calculateOutputSize(t),o=new Array(this.raggedRank+1);o[o.length-1]=1;for(let a=o.length-2;a>=0;--a)o[a]=o[a+1]*s[a+1];const l=lt(s,!1),r=R(this.valuesDType,S(l));if(o[0]*s[0]>0){let a=this.calculateFirstParentOutputIndex(t,o[0],s[0]);for(let i=1;i<=this.raggedRank;++i)a=this.calculateOutputIndex(i-1,a,o[i],s[i]);this.setOutput(this.raggedRank,a,r,l)}return[l,r]}setOutput(e,t,s,o){if(s.length===0)return;const l=this.values,r=s;let c=o.slice();c=c.slice(e+1);const a=S(c),i=t.length;let f=this.defaultValue;if(f.length!==a&&f.length!==1){const d=this.defaultValueShape;Ce(()=>{const w=Fe(f,d);f=Ve(w,c).dataSync()})}let u=0,g=0,m=0;for(let d=0;d<=i;++d){let w=d<i?t[d]:-1;if(w===m){++m;continue}if(g<m){const h=l.subarray(u*a),p=r.subarray(g*a),I=(m-g)*a;rt(p,h,I)}if(d>=i){const h=s.length;w=Math.floor(h/a)}if(w>m)if(this.defaultValue.length===1)r.subarray(m*a,w*a).fill(this.defaultValue[0]),m=w;else for(;w>m;){const h=r.slice(m*a);rt(h,f,a),++m}w<0?(u=d+1,g=m):(u=d,g=m,m=g+1)}}}function rt(n,e,t){for(let s=0;s<t;s++)n[s]=e[s]}function lt(n,e){const t=[];for(let s of n){if(s<0){if(!e)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}t.push(s)}return t}function On(n,e,t,s,o,l,r,c,a,i){return new B(n,e,t,s,o,l,r,c,a,i).compute()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qn(n,e,t,s){const o=n===e,l=n<e&&t<0,r=e<n&&t>1;if(o||l||r)return O(0,s);const c=Math.abs(Math.ceil((e-n)/t)),a=O(c,s);e<n&&t===1&&(t=-1),a[0]=n;for(let i=1;i<a.length;i++)a[i]=a[i-1]+t;return a}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oe=F(n=>1/Math.sqrt(n)),Ln=V(Nt,oe),Vs={kernelName:Nt,backendName:"cpu",kernelFunc:Ln};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $n(n,e,t,s,o,l,r,c,a,i){const f=[s/o,o],u=n.values,g=e.values;if(s===0)return P(t,e.dtype);const m=a instanceof X?a:P(f,e.dtype);typeof a=="string"||typeof a=="number"?m.values.fill(a):typeof a=="boolean"&&m.values.fill(+a);for(let d=0;d<l;d++){const w=[];let h=0;for(let p=0;p<r;p++){const I=u[d*r+p];w.push(I),h+=I*c[p]}if(h<0||h>=s/o)throw new Error(`Invalid indices: ${w} does not index into ${t}`);for(let p=0;p<o;p++)i?m.values[h*o+p]+=g[d*o+p]:m.values[h*o+p]=e.rank===0?g[0]:g[d*o+p]}return m}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zn=F(n=>1/(1+Math.exp(-n))),Gn=$t(Mt,n=>1/(1+Math.exp(-n))),Ds={kernelName:Mt,backendName:"cpu",kernelFunc:Gn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ae(n,e,t,s,o){const l=De(s,e,t),r=S(t),c=C(s);if(l){const u=Ae(e,c);return o==="string"?n.slice(u,u+r):n.subarray(u,u+r)}const a=o==="string"?G(n):n,i=P(s,o,a),f=P(t,o);for(let u=0;u<f.size;++u){const g=f.indexToLoc(u),m=g.map((d,w)=>d+e[w]);f.set(i.get(...m),...g)}return o==="string"?Oe(f.values):f.values}function Wn(n){const{inputs:e,backend:t,attrs:s}=n,{x:o}=e,{begin:l,size:r}=s;A(o,"slice");const[c,a]=Le(o,l,r);$e(o,c,a);const i=t.data.get(o.dataId).values,f=ae(i,c,a,o.shape,o.dtype);return t.makeTensorInfo(a,o.dtype,f)}const As={kernelName:qe,backendName:"cpu",kernelFunc:Wn};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jn(n,e,t,s,o,l,r){const c=e[0],a=l[0],i=new Array(a),f=new Array(c),u=e[1];if(a===0){if(c!==0)throw new Error(ze(c));const h=R(t,0),p=R(o,0);return[h,[0,u],p,i,f]}let g=!0,m=0;const d=new Array(a).fill(0);for(let h=0;h<c;++h){const p=n[h*u];if(p<0)throw new Error(Ge(h,p));if(p>=a)throw new Error(We(h,p,a));++d[p],g=g&&p>=m,m=p}let w=!0;for(let h=0;h<a;++h){const p=d[h]===0;i[h]=p,w=w&&!p,d[h]=Math.max(d[h],1),h>0&&(d[h]+=d[h-1])}if(w&&g){const h=n,p=s;for(let I=0;I<c;++I)f[I]=I;return[h,[c,u],p,i,f]}else{const h=d[a-1],p=R(t,h*u),I=R(o,h),k=new Array(a).fill(0);for(let x=0;x<c;++x){const b=n[x*u],y=k[b],E=(b===0?0:d[b-1])+y;k[b]++;for(let N=0;N<u;++N)p[E*u+N]=n[x*u+N];I[E]=s[x],f[x]=E}for(let x=0;x<a;++x)if(k[x]===0){const y=x===0?0:d[x-1];p[y*u+0]=x;for(let E=1;E<u;++E)p[y*u+E]=0;I[y]=r}return[p,[h,u],I,i,f]}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _n(n,e,t,s,o){const l=S(s),r=e[0],c=o.length,a=[];let i=1,f=-1;for(let h=0;h<c;++h){const p=o[h];if(p===-1){if(f!==-1)throw new Error(je(f,h));f=h,a.push(1)}else{if(p<0)throw new Error(_e(h,p));i*=p,a.push(p)}}if(f!==-1){if(i<=0)throw new Error(Be());const h=Math.trunc(l/i);if(i*h!==l)throw new Error(Ke(s,a));a[f]=h}if(S(a)!==l)throw new Error(Ue(s,a));const g=s.length,m=[];if(g>0){m[g-1]=1;for(let h=g-2;h>=0;--h)m[h]=m[h+1]*s[h+1]}const d=[];if(c>0){d[c-1]=1;for(let h=c-2;h>=0;--h)d[h]=d[h+1]*a[h+1]}const w=R(t,r*c);for(let h=0;h<r;++h){let p=0;for(let I=0;I<g;++I)p+=n[h*g+I]*m[I];for(let I=0;I<c;++I)w[h*c+I]=Math.trunc(p/d[I]),p%=d[I]}return[w,[r,c],a]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bn(n,e,t,s,o,l=!1,r=0){const c=s.length,a=[e[0],n.length/e[0]],i=a[1],u=c>0?o[c-1]+1:0;if(u<0)throw new Error(st());const g=e.slice();g[0]=u;const m=g.reduce((k,x)=>k*x,1),d=R(t,m);if(c===0)return u>0&&d.fill(r),[d,g];if(u<=0)throw new Error(st());let w=0,h=1,p=0,I=o[w];for(;;){let k=0;if(h<c){if(k=o[h],I===k){++h;continue}if(I>=k)throw new Error(Ze())}if(I<0||I>=u)throw new Error(He(I,u));I>p&&d.fill(r,p*i,I*i);for(let x=w;x<h;++x){const b=s[x];if(b<0||b>=a[0])throw new Error(Xe(x,s[x],a[0]));for(let y=0;y<i;y++)d[I*i+y]+=n[b*i+y]}if(l)for(let x=0;x<i;x++)d[I*i+x]/=h-w;if(w=h,++h,p=I+1,I=k,h>c)break}return p<u&&d.fill(r,p*i,u*i),[d,g]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kn=F(n=>Math.sqrt(n)),Un=$t(Pt,n=>Math.sqrt(n)),Os={kernelName:Pt,backendName:"cpu",kernelFunc:Un};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const re=v(((n,e)=>{const t=n-e;return t*t})),Zn=T(Ct,re),qs={kernelName:Ct,backendName:"cpu",kernelFunc:Zn};/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const le=F((n,e)=>{const{pattern:t,replaceGlobal:s,rewrite:o}=e;return n.replace(new RegExp(t,s?"g":""),o)}),Hn=V(Ft,le),Ls={kernelName:Ft,backendName:"cpu",kernelFunc:Hn};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xn(n,e,t,s){const o=P(n,e.dtype);for(let l=0;l<o.size;l++){const r=o.indexToLoc(l),c=new Array(r.length);for(let a=0;a<c.length;a++)c[a]=r[a]*t[a]+s[a];o.set(e.get(...c),...r)}return o}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Qn{constructor(e,t,s,o,l,r){this.separator=H(e),this.nGramWidths=t,this.leftPad=H(s),this.rightPad=H(o),this.padWidth=l,this.preserveShort=r}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const s=this.getPadWidth(t);return Math.max(0,e+2*s-t+1)}createNGrams(e,t,s,o,l,r){for(let c=0;c<l;++c){const a=this.getPadWidth(r),i=Math.max(0,a-c),f=Math.max(0,a-(l-(c+1))),u=r-(i+f),g=t+(i>0?0:c-a);let m=0;m+=i*this.leftPad.length;for(let I=0;I<u;++I)m+=e[g+I].length;m+=f*this.rightPad.length;const d=i+f+u-1;m+=d*this.separator.length,s[o+c]=new Uint8Array(m);const w=s[o+c];let h=0;const p=I=>I.forEach(k=>w[h++]=k);for(let I=0;I<i;++I)p(this.leftPad),p(this.separator);for(let I=0;I<u-1;++I)p(e[g+I]),p(this.separator);if(u>0){p(e[g+u-1]);for(let I=0;I<f;++I)p(this.separator),p(this.rightPad)}else{for(let I=0;I<f-1;++I)p(this.rightPad),p(this.separator);p(this.rightPad)}}}compute(e,t){const s=e.length,o=t.length;if(o>0){let a=t[0];if(a!==0)throw new Error(`First split value must be 0, got ${a}`);for(let i=1;i<o;++i){let f=t[i]>=a;if(f=f&&t[i]<=s,!f)throw new Error(`Invalid split value ${t[i]}, must be in [${a}, ${s}]`);a=t[i]}if(a!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${a}`)}const l=o-1,r=R("int32",o);if(s===0||o===0){const a=new Array(s);for(let i=0;i<=l;++i)r[i]=0;return[a,r]}r[0]=0;for(let a=1;a<=l;++a){const i=t[a]-t[a-1];let f=0;this.nGramWidths.forEach(u=>{f+=this.getNumNGrams(i,u)}),this.preserveShort&&i>0&&f===0&&(f=1),r[a]=r[a-1]+f}const c=new Array(r[l]);for(let a=0;a<l;++a){const i=t[a];let f=r[a];if(this.nGramWidths.forEach(u=>{const g=t[a+1]-t[a],m=this.getNumNGrams(g,u);this.createNGrams(e,i,c,f,m,u),f+=m}),this.preserveShort&&f===r[a]){const u=t[a+1]-t[a];if(u===0)continue;const g=u+2*this.padWidth;this.createNGrams(e,i,c,f,1,g)}}return[c,r]}}function Yn(n,e,t,s,o,l,r,c){return new Qn(t,s,o,l,r,c).compute(n,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jn(n,e,t,s){if(!n.length)return;if(e.length===0){for(let l=0;l<n.length;++l)s.push(n.subarray(l,l+1));return}if(e.length===1){const l=e[0];let r=n.indexOf(l);for(;r!==-1;){const c=n.subarray(0,r);(!t||c.length!==0)&&s.push(c),n=n.subarray(r+1),r=n.indexOf(l)}(!t||n.length!==0)&&s.push(n);return}let o=0;for(let l=0;l<n.length+1;l++)if(l===n.length||e.indexOf(n[l])!==-1){const r=n.subarray(o,l);(!t||r.length!==0)&&s.push(r),o=l+1}}function ts(n,e,t){const s=n.length,o=[];let l=0,r=0;const c=new Array(s);for(let g=0;g<s;++g){const m=o.length;Jn(n[g],e,t,o);const d=o.length-m;c[g]=d,l+=d,r=Math.max(r,d)}const a=R("int32",l*2),i=new Array(l),f=[s,r];let u=0;for(let g=0;g<s;++g)for(let m=0;m<c[g];++m)a[u*2]=g,a[u*2+1]=m,i[u]=o[u],++u;return[a,i,f]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function es(n,e){const t=R("int32",n.length);for(let s=0;s<n.length;++s)t[s]=Qe(n[s]).modulo(e).getLowBitsUnsigned();return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ce=v(((n,e)=>n-e)),ns=J(((n,e,t,s)=>({real:n-t,imag:e-s}))),ss=T(Vt,ce,ns),$s={kernelName:Vt,backendName:"cpu",kernelFunc:ss};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function os(n,e){const t=new Array(n.rank);for(let o=0;o<t.length;o++)t[o]=n.shape[o]*e[o];const s=P(t,n.dtype);for(let o=0;o<s.values.length;++o){const l=s.indexToLoc(o),r=new Array(n.rank);for(let a=0;a<r.length;a++)r[a]=l[a]%n.shape[a];const c=n.locToIndex(r);s.values[o]=n.values[c]}return s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $=(n,e)=>{const t=e.value-n.value;return t===0?n.index-e.index:t};function ie(n,e,t=0,s=n.length-1){for(;s>t;){if(s-t>600){const c=s-t+1,a=e-t+1,i=Math.log(c),f=.5*Math.exp(2*i/3),u=.5*Math.sqrt(i*f*(c-f)/c)*Math.sign(a-c/2),g=Math.max(t,Math.floor(e-a*f/c+u)),m=Math.min(s,Math.floor(e+(c-a)*f/c+u));ie(n,e,g,m)}const o=n[e];let l=t,r=s;for(L(n,t,e),$(n[s],o)>0&&L(n,t,s);l<r;){for(L(n,l,r),l++,r--;$(n[l],o)<0;)l=l+1;for(;$(n[r],o)>0;)r=r-1}$(n[t],o)===0?L(n,t,r):(r=r+1,L(n,r,s)),r<=e&&(t=r+1),e<=r&&(s=r-1)}}function as(n,e,t,s,o){const l=e[e.length-1],[r,c]=[n.length/l,l],a=D(t,r*s),i=D("int32",r*s);for(let u=0;u<r;u++){const g=u*c,m=n.subarray(g,g+c);let d=new Array(m.length);m.forEach((I,k)=>d[k]={value:I,index:k}),s<d.length&&(ie(d,s),d=d.slice(0,s)),o&&d.sort($);const w=u*s,h=a.subarray(w,w+s),p=i.subarray(w,w+s);for(let I=0;I<s;I++)h[I]=d[I].value,p[I]=d[I].index}const f=e.slice();return f[f.length-1]=s,[P(f,t,a),P(f,"int32",i)]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rs(n,e,t,s){const o=Tt(e,t)[0],l=[1,t[0],1];for(let d=0;d<o;d++)l[0]*=t[d];l[1]=t[o];for(let d=o+1;d<t.length;d++)l[2]*=t[d];const r=new Map,c=new Int32Array(t[o]),a=new X(l,s,n),i=[],f=l[0]===1&&l[2]===1;for(let d=0;d<t[o];d++){let w;if(f)w=n[d].toString();else{const p=[];for(let I=0;I<l[0];I++)for(let k=0;k<l[2];k++)p.push(a.get(I,d,k));w=p.join(",")}const h=r.get(w);if(h!=null)c[d]=h;else{const p=r.size;r.set(w,p),c[d]=p,i.push(d)}}const u=l.slice();u[1]=r.size;const g=new X(u,s);i.forEach((d,w)=>{for(let h=0;h<l[0];h++)for(let p=0;p<l[2];p++)g.set(a.get(h,d,p),h,w,p)});const m=t.slice();return m[o]=u[1],{outputValues:g.values,outputShape:m,indices:c}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zs=Object.freeze(Object.defineProperty({__proto__:null,addImpl:qt,bincountImpl:en,bincountReduceImpl:nn,bitwiseAndImpl:Lt,castImpl:Ot,ceilImpl:zt,concatImpl:an,equalImpl:Gt,expImpl:Wt,expm1Impl:jt,floorDivImpl:Bt,floorImpl:_t,gatherNdImpl:hn,gatherV2Impl:dn,greaterEqualImpl:Ut,greaterImpl:Kt,lessEqualImpl:Ht,lessImpl:Zt,linSpaceImpl:wn,logImpl:Xt,maxImpl:kn,maximumImpl:Qt,minimumImpl:Yt,multiplyImpl:tt,negImpl:Jt,notEqualImpl:te,prodImpl:se,raggedGatherImpl:Dn,raggedRangeImpl:An,raggedTensorToTensorImpl:On,rangeImpl:qn,rsqrtImpl:oe,scatterImpl:$n,sigmoidImpl:zn,simpleAbsImpl:Dt,sliceImpl:ae,sparseFillEmptyRowsImpl:jn,sparseReshapeImpl:_n,sparseSegmentReductionImpl:Bn,sqrtImpl:Kn,squaredDifferenceImpl:re,staticRegexReplaceImpl:le,stridedSliceImpl:Xn,stringNGramsImpl:Yn,stringSplitImpl:ts,stringToHashBucketFastImpl:es,subImpl:ce,tileImpl:os,topKImpl:as,transposeImpl:ee,uniqueImpl:rs},Symbol.toStringTag,{value:"Module"}));export{ys as $,An as A,On as B,qn as C,$n as D,jn as E,_n as F,Bn as G,Xn as H,Yn as I,ts as J,es as K,os as L,as as M,rs as N,rn as O,cs as P,ds as Q,ps as R,hs as S,gs as T,is as U,ms as V,Is as W,ws as X,xs as Y,ks as Z,bs as _,A as a,us as a0,Es as a1,Ss as a2,Rs as a3,vs as a4,Ts as a5,Ns as a6,Ms as a7,Ps as a8,Fs as a9,fs as aa,Vs as ab,Ds as ac,As as ad,Os as ae,qs as af,Ls as ag,$s as ah,Cs as ai,Gn as b,v as c,tn as d,T as e,Wn as f,en as g,U as h,Y as i,an as j,nn as k,W as l,Sn as m,ss as n,hn as o,dn as p,wn as q,At as r,zs as s,ne as t,$t as u,ee as v,kn as w,ln as x,Dn as y,Q as z};
