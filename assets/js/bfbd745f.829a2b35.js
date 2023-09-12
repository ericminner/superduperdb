"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[482],{3905:(e,t,r)=>{r.d(t,{Zo:()=>c,kt:()=>h});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function l(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},a=Object.keys(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(n=0;n<a.length;n++)r=a[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var s=n.createContext({}),u=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):i(i({},t),e)),r},c=function(e){var t=u(e.components);return n.createElement(s.Provider,{value:t},e.children)},p="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,a=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),p=u(r),m=o,h=p["".concat(s,".").concat(m)]||p[m]||d[m]||a;return r?n.createElement(h,i(i({ref:t},c),{},{components:r})):n.createElement(h,i({ref:t},c))}));function h(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=r.length,i=new Array(a);i[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[p]="string"==typeof e?e:o,i[1]=l;for(var u=2;u<a;u++)i[u]=r[u];return n.createElement.apply(null,i)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},1425:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>d,frontMatter:()=>a,metadata:()=>l,toc:()=>u});var n=r(7462),o=(r(7294),r(3905));const a={sidebar_position:1},i="Overview of SuperDuperDB clusters",l={unversionedId:"docs/cluster/intro",id:"docs/cluster/intro",title:"Overview of SuperDuperDB clusters",description:"There are 3 features of a SuperDuperDB cluster:",source:"@site/content/docs/cluster/intro.md",sourceDirName:"docs/cluster",slug:"/docs/cluster/intro",permalink:"/docs/docs/cluster/intro",draft:!1,editUrl:"https://github.com/SuperDuperDB/superduperdb/content/docs/cluster/intro.md",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"Cluster",permalink:"/docs/category/cluster"},next:{title:"SuperDuperDB architecture",permalink:"/docs/docs/cluster/architecture"}},s={},u=[{value:"Components",id:"components",level:2},{value:"Client",id:"client",level:3},{value:"Datastore",id:"datastore",level:3},{value:"Linear algebra",id:"linear-algebra",level:3},{value:"Model-server",id:"model-server",level:3},{value:"Worker",id:"worker",level:3}],c={toc:u},p="wrapper";function d(e){let{components:t,...a}=e;return(0,o.kt)(p,(0,n.Z)({},c,a,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"overview-of-superduperdb-clusters"},"Overview of SuperDuperDB clusters"),(0,o.kt)("p",null,"There are 3 features of a SuperDuperDB cluster:"),(0,o.kt)("ol",null,(0,o.kt)("li",{parentName:"ol"},"When users access SuperDuperDB, they invoke the SuperDuperDB ",(0,o.kt)("strong",{parentName:"li"},"client")," which manages communication\nwith the various components of the SuperDuperDB cluster."),(0,o.kt)("li",{parentName:"ol"},"Data storage: occurs in the underlying datastore, where the raw data,\n","[models]","(Models - an extension of PyTorch models) and model outputs are stored."),(0,o.kt)("li",{parentName:"ol"},"Job management: whenever SuperDuperDB creates a ","[job]","(Jobs - scheduling of training and model outputs)\n(i.e. during data inserts, updates, downloads, and model training), a job is queued and\nthen executed by a pool of ",(0,o.kt)("strong",{parentName:"li"},"workers"),"."),(0,o.kt)("li",{parentName:"ol"},"Querying SuperDuperDB: when using queries which use tensor similarity, the SuperDuperDB client\ncombines calls to the datastore with calls to the ",(0,o.kt)("strong",{parentName:"li"},"vector-search")," component."),(0,o.kt)("li",{parentName:"ol"},"SuperDuperDB includes a ",(0,o.kt)("strong",{parentName:"li"},"model-server"),", which may be used to serve the models which have\nbeen uploaded to SuperDuperDB.")),(0,o.kt)("p",null,"The exact setup of your SuperDuperDB cluster will depend on the use-cases you\nplan to execute with the cluster. Factors to consider will be:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Latency"),(0,o.kt)("li",{parentName:"ul"},"Where the MongoDB deployment is located"),(0,o.kt)("li",{parentName:"ul"},"Whether you want to scale the cluster according to demand"),(0,o.kt)("li",{parentName:"ul"},"What hardware you'd like to run"),(0,o.kt)("li",{parentName:"ul"},"And more...")),(0,o.kt)("h2",{id:"components"},"Components"),(0,o.kt)("p",null,"The basic topology of a SuperDuperDB cluster is given in the graphic below:"),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"SuperDuperDB cluster topology",src:r(6340).Z,width:"2172",height:"1380"})),(0,o.kt)("h3",{id:"client"},"Client"),(0,o.kt)("p",null,"This is the programmer's interface to\nthe SuperDuperDB cluster and provides a unified user-experience very similar to the underlying datastore's\nuser experience."),(0,o.kt)("h3",{id:"datastore"},"Datastore"),(0,o.kt)("p",null,"This is your standard datastore deployment. The deployment can either sit in the same infrastructure\nas the remainder of the SuperDuperDB cluster, or it can be situated remotely. Performance and latency\nconcerns here will play a role in which version works best and is most convenient."),(0,o.kt)("h3",{id:"linear-algebra"},"Linear algebra"),(0,o.kt)("p",null,"This node returns real time semantic index search outputs to the client. The node loads\nmodel outputs which are of vector or tensor type, and creates an in-memory search index over\nthem."),(0,o.kt)("h3",{id:"model-server"},"Model-server"),(0,o.kt)("p",null,"SuperDuperDB contains a component which serves models which has been created."),(0,o.kt)("h3",{id:"worker"},"Worker"),(0,o.kt)("p",null,"These nodes perform the long computations necessary to update model outputs when new data\ncome in, and also perform model training for models which are set up to be trained on creation."))}d.isMDXComponent=!0},6340:(e,t,r)=>{r.d(t,{Z:()=>n});const n=r.p+"assets/images/architecture_now-7c1983550e004665884590286ebe79ab.png"}}]);