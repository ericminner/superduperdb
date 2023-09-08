"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[7302],{3905:(e,n,t)=>{t.d(n,{Zo:()=>u,kt:()=>f});var r=t(7294);function o(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function p(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){o(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function s(e,n){if(null==e)return{};var t,r,o=function(e,n){if(null==e)return{};var t,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(o[t]=e[t])}return o}var c=r.createContext({}),i=function(e){var n=r.useContext(c),t=n;return e&&(t="function"==typeof e?e(n):p(p({},n),e)),t},u=function(e){var n=i(e.components);return r.createElement(c.Provider,{value:n},e.children)},l="mdxType",d={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,o=e.mdxType,a=e.originalType,c=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),l=i(t),m=o,f=l["".concat(c,".").concat(m)]||l[m]||d[m]||a;return t?r.createElement(f,p(p({ref:n},u),{},{components:t})):r.createElement(f,p({ref:n},u))}));function f(e,n){var t=arguments,o=n&&n.mdxType;if("string"==typeof e||o){var a=t.length,p=new Array(a);p[0]=m;var s={};for(var c in n)hasOwnProperty.call(n,c)&&(s[c]=n[c]);s.originalType=e,s[l]="string"==typeof e?e:o,p[1]=s;for(var i=2;i<a;i++)p[i]=t[i];return r.createElement.apply(null,p)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},7901:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>p,default:()=>d,frontMatter:()=>a,metadata:()=>s,toc:()=>i});var r=t(7462),o=(t(7294),t(3905));const a={},p="Ask the docs anything about SuperDuperDB",s={unversionedId:"use_cases/question-the-docs",id:"use_cases/question-the-docs",title:"Ask the docs anything about SuperDuperDB",description:"",source:"@site/content/use_cases/question-the-docs.md",sourceDirName:"use_cases",slug:"/use_cases/question-the-docs",permalink:"/docs/use_cases/question-the-docs",draft:!1,editUrl:"https://github.com/SuperDuperDB/superduperdb/content/use_cases/question-the-docs.md",tags:[],version:"current",frontMatter:{},sidebar:"useCasesSidebar",previous:{title:"OpenAI vector search",permalink:"/docs/use_cases/openai"},next:{title:"Creating a DB of image features in torchvision",permalink:"/docs/use_cases/resnet_features"}},c={},i=[],u={toc:i},l="wrapper";function d(e){let{components:n,...t}=e;return(0,o.kt)(l,(0,r.Z)({},u,t,{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"ask-the-docs-anything-about-superduperdb"},"Ask the docs anything about SuperDuperDB"),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import os\nos.environ['OPENAI_API_KEY'] = '<YOUR-OPENAI-API-KEY>'\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb import superduper\nfrom superduperdb.db.mongodb.query import Collection\nimport pymongo\n\ndb = pymongo.MongoClient().documents\ndb = superduper(db)\n\ncollection = Collection('questiondocs')\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"import glob\n\nSTRIDE = 5       # stride in numbers of lines\nWINDOW = 10       # length of window in numbers of lines\n\ncontent = sum([open(file).readlines() for file in glob.glob('../*/*.md') + glob.glob('../*.md')], [])\nchunks = ['\\n'.join(content[i: i + WINDOW]) for i in range(0, len(content), STRIDE)]\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from IPython.display import Markdown\nMarkdown(chunks[2])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.document import Document\n\ndb.execute(collection.insert_many([Document({'txt': chunk}) for chunk in chunks]))\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"db.execute(collection.find_one())\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.vector_index import VectorIndex\nfrom superduperdb.container.listener import Listener\nfrom superduperdb.ext.openai.model import OpenAIEmbedding\n\ndb.add(\n    VectorIndex(\n        identifier='my-index',\n        indexing_listener=Listener(\n            model=OpenAIEmbedding(model='text-embedding-ada-002'),\n            key='txt',\n            select=collection.find(),\n        ),\n    )\n)\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.ext.openai.model import OpenAIChatCompletion\n\nchat = OpenAIChatCompletion(\n    model='gpt-3.5-turbo',\n    prompt=(\n        'Use the following description and code-snippets aboout SuperDuperDB to answer this question about SuperDuperDB\\n'\n        'Do not use any other information you might have learned about other python packages\\n'\n        'Only base your answer on the code-snippets retrieved\\n'\n        '{context}\\n\\n'\n        'Here\\'s the question:\\n'\n    ),\n)\n\ndb.add(chat)\n\nprint(db.show('model'))\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"db.show('model', 'gpt-3.5-turbo')\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.document import Document\nfrom IPython.display import display, Markdown\n\n\nq = 'Can you give me a code-snippet to set up a `VectorIndex`?'\n\noutput, context = db.predict(\n    model='gpt-3.5-turbo',\n    input=q,\n    context_select=(\n        collection\n            .like(Document({'txt': q}), vector_index='my-index', n=5)\n            .find()\n    ),\n    context_key='txt',\n)\n\nMarkdown(output.content)\n")))}d.isMDXComponent=!0}}]);