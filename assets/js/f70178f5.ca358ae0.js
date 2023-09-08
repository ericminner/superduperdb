"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[6727],{3905:(e,n,r)=>{r.d(n,{Zo:()=>d,kt:()=>f});var t=r(7294);function o(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function i(e,n){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);n&&(t=t.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),r.push.apply(r,t)}return r}function a(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{};n%2?i(Object(r),!0).forEach((function(n){o(e,n,r[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(r,n))}))}return e}function s(e,n){if(null==e)return{};var r,t,o=function(e,n){if(null==e)return{};var r,t,o={},i=Object.keys(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||(o[r]=e[r]);return o}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var l=t.createContext({}),c=function(e){var n=t.useContext(l),r=n;return e&&(r="function"==typeof e?e(n):a(a({},n),e)),r},d=function(e){var n=c(e.components);return t.createElement(l.Provider,{value:n},e.children)},p="mdxType",u={inlineCode:"code",wrapper:function(e){var n=e.children;return t.createElement(t.Fragment,{},n)}},m=t.forwardRef((function(e,n){var r=e.components,o=e.mdxType,i=e.originalType,l=e.parentName,d=s(e,["components","mdxType","originalType","parentName"]),p=c(r),m=o,f=p["".concat(l,".").concat(m)]||p[m]||u[m]||i;return r?t.createElement(f,a(a({ref:n},d),{},{components:r})):t.createElement(f,a({ref:n},d))}));function f(e,n){var r=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var i=r.length,a=new Array(i);a[0]=m;var s={};for(var l in n)hasOwnProperty.call(n,l)&&(s[l]=n[l]);s.originalType=e,s[p]="string"==typeof e?e:o,a[1]=s;for(var c=2;c<i;c++)a[c]=r[c];return t.createElement.apply(null,a)}return t.createElement.apply(null,r)}m.displayName="MDXCreateElement"},7218:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>l,contentTitle:()=>a,default:()=>u,frontMatter:()=>i,metadata:()=>s,toc:()=>c});var t=r(7462),o=(r(7294),r(3905));const i={sidebar_position:5},a="Vector Indexes",s={unversionedId:"docs/usage/vector_index",id:"docs/usage/vector_index",title:"Vector Indexes",description:"SuperDuperDB has support for vector-search via LanceDB using vector-indexes.",source:"@site/content/docs/usage/vector_index.md",sourceDirName:"docs/usage",slug:"/docs/usage/vector_index",permalink:"/docs/docs/usage/vector_index",draft:!1,editUrl:"https://github.com/SuperDuperDB/superduperdb/content/docs/usage/vector_index.md",tags:[],version:"current",sidebarPosition:5,frontMatter:{sidebar_position:5},sidebar:"tutorialSidebar",previous:{title:"Queries",permalink:"/docs/docs/usage/queries"},next:{title:"Datasets",permalink:"/docs/docs/usage/datasets"}},l={},c=[{value:"Creating vector indexes",id:"creating-vector-indexes",level:2},{value:"Using vector indexes",id:"using-vector-indexes",level:2},{value:"MongoDB",id:"mongodb",level:3}],d={toc:c},p="wrapper";function u(e){let{components:n,...r}=e;return(0,o.kt)(p,(0,t.Z)({},d,r,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"vector-indexes"},"Vector Indexes"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-{note}"},"SuperDuperDB provides first-class support for Vector-Search, including \nencoding of inputs by arbitrary AI models.\n")),(0,o.kt)("p",null,"SuperDuperDB has support for vector-search via LanceDB using vector-indexes.\nWe are working on support for vector-search via MongoDB enterprise search in parallel."),(0,o.kt)("p",null,"Vector-indexes build on top of the ",(0,o.kt)("a",{parentName:"p",href:"db"},"DB"),", ",(0,o.kt)("a",{parentName:"p",href:"models"},"models")," and ",(0,o.kt)("a",{parentName:"p",href:"listeners"},"listeners"),"."),(0,o.kt)("h2",{id:"creating-vector-indexes"},"Creating vector indexes"),(0,o.kt)("p",null,"In order to build a vector index, one defines one or two models, and daemonizes them with listeners.\nIn the simples variant one does simply:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.vector_index import VectorIndex\nfrom sueprduperdb.core.listener import listener\n\ndb.add(\n    VectorIndex(indexing_listener='my-model/my-key')\n)\n")),(0,o.kt)("p",null,"The model ",(0,o.kt)("inlineCode",{parentName:"p"},"my-model")," should have already been registered with SuperDuperDB (see ",(0,o.kt)("a",{parentName:"p",href:"models"},"models")," for help). ",(0,o.kt)("inlineCode",{parentName:"p"},"my-key")," is the field to be searched. Together ",(0,o.kt)("inlineCode",{parentName:"p"},"my-model/my-key")," refer to the ",(0,o.kt)("a",{parentName:"p",href:"listeners"},"listener")," component (previously created) which is responsible for computing vectors from the data.\nSee ",(0,o.kt)("a",{parentName:"p",href:"listener"},"here")," for how to create such a component."),(0,o.kt)("p",null,"Alternatively the model and listener may be created inline.\nHere is how to define a simple linear bag-of-words model:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.vector_index import VectorIndex\nfrom superduperdb.container.listener import Listener\nfrom superduperdb.model.sentence_transformers.wrapper import Pipeline\n\n\nclass TextEmbedding:\n    def __init__(self, lookup):\n        self.lookup = lookup  # mapping from strings to pytorch tensors\n\n    def __call__(self, x):\n        return sum([self.lookup[y] for y in x.split()])\n\n for\ndb.add(\n    VectorIndex(\n        identifier='my-index',\n        indexing_listener=Listener(\n            model=TorchModel(\n                preprocess=TextEmbedding(d),  # \"d\" should be loaded from disk\n                object=torch.nn.Linear(64, 512),\n            )\n        key = '<key-to-search>',\n    )\n)\n")),(0,o.kt)("h2",{id:"using-vector-indexes"},"Using vector indexes"),(0,o.kt)("h3",{id:"mongodb"},"MongoDB"),(0,o.kt)("p",null,"To use your vector index to search MongoDB, there are two possibilities:"),(0,o.kt)("p",null,"Firstly, find similar matches and then filter the results:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},">>> from superduperdb.container.document import Document as D\n>>> db.execute(\n...    Collection('my-coll')\n...       .like(D({'<key-to-search>': '<content' >}, vector_index='my-index')\n...       .find( < filter >, < projection >)\n...    )\n... )\n")),(0,o.kt)("p",null,"Secondly, filter the data and find similar matches within the results:"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},">>> db.execute(\n...    Collection('my-coll')\n...        .like(D({'<key-to-search>': '<content'>}), vector_index='my-index')\n...        .find(<filter>, <projection>)\n... )\n")))}u.isMDXComponent=!0}}]);