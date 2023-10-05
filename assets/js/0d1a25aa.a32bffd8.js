"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[2163],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>h});var r=n(67294);function o(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){o(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,o=function(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var l=r.createContext({}),p=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(l.Provider,{value:t},e.children)},c="mdxType",d={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,o=e.mdxType,a=e.originalType,l=e.parentName,u=s(e,["components","mdxType","originalType","parentName"]),c=p(n),m=o,h=c["".concat(l,".").concat(m)]||c[m]||d[m]||a;return n?r.createElement(h,i(i({ref:t},u),{},{components:n})):r.createElement(h,i({ref:t},u))}));function h(e,t){var n=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var a=n.length,i=new Array(a);i[0]=m;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s[c]="string"==typeof e?e:o,i[1]=s;for(var p=2;p<a;p++)i[p]=n[p];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},32576:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>i,default:()=>d,frontMatter:()=>a,metadata:()=>s,toc:()=>p});var r=n(87462),o=(n(67294),n(3905));const a={},i="The easiest way to implement question-your documents we know",s={permalink:"/blog/2023/10/04/walkthrough-rag-app-atlas",editUrl:"https://github.com/SuperDuperDB/superduperdb/tree/main/docs/blog/2023-10-04-walkthrough-rag-app-atlas.md",source:"@site/blog/2023-10-04-walkthrough-rag-app-atlas.md",title:"The easiest way to implement question-your documents we know",description:"*Despite the huge surge in popularity in building AI applications with LLMs and vector-search,",date:"2023-10-04T00:00:00.000Z",formattedDate:"October 4, 2023",tags:[],readingTime:2.65,hasTruncateMarker:!0,authors:[],frontMatter:{},nextItem:{title:"A walkthrough of vector-search on MongoDB Atlas with SuperDuperDB",permalink:"/blog/2023/09/31/a-walkthrough-of-vector-search-on-mongodb-atlas-with-superduperdb/content"}},l={authorsImageUrls:[]},p=[],u={toc:p},c="wrapper";function d(e){let{components:t,...n}=e;return(0,o.kt)(c,(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,(0,o.kt)("em",{parentName:"p"},"Despite the huge surge in popularity in building AI applications with LLMs and vector-search,\nwe haven't seen any walkthroughs boil this down to a super-simple, few-command process.\nWith SuperDuperDB together with MongoDB Atlas, it's easier and more flexible than ever before.")),(0,o.kt)("hr",null),(0,o.kt)("p",null,"Setting up a question-your-documents service can be a complex process."),(0,o.kt)("p",null,"There are several steps involved in doing this:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"Serve a model or forward requests to convert text-data in the database to vectors in a vector-database"),(0,o.kt)("li",{parentName:"ul"},"Setting up a vector-index in a vector-database which efficiently finds similar vectors"),(0,o.kt)("li",{parentName:"ul"},"Setting up an endpoint to either run a self hosted LLM  or forward requests to a question-answering LLM such as OpenAI"),(0,o.kt)("li",{parentName:"ul"},"Setting up an endpoint to:",(0,o.kt)("ul",{parentName:"li"},(0,o.kt)("li",{parentName:"ul"},"Convert a question to a vector"),(0,o.kt)("li",{parentName:"ul"},"Find relevant documents to the question using vector-search"),(0,o.kt)("li",{parentName:"ul"},"Send the documents as context to the question-answering LLM")))),(0,o.kt)("p",null,"This process can be tedious and complex, involving several pieces of infrastructure, especially\nif developers would like to use other models than those hosted behind OpenAI's API."),(0,o.kt)("p",null,"What if we told you that with SuperDuperDB together with MongoDB Atlas, these challenges are a thing of the past,\nand can be done more simply than with any other solution available?"),(0,o.kt)("p",null,"Let's dive straight into the solution:"),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Connect to MongoDB Atlas with SuperDuperDB")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},'from superduperdb.db.base.build import build_datalayer\nfrom superduperdb import CFG\nimport os\n\nATLAS_URI = "mongodb+srv://<user>@<atlas-server>/<database_name>"\nOPENAI_API_KEY = "<your-open-ai-api-key>"\n\nos.environ["OPENAI_API_KEY"] = OPENAI_API_KEY\n\nCFG.data_backend = ATLAS_URI\nCFG.vector_search = ATLAS_URI\n\ndb = build_datalayer()\n')),(0,o.kt)("p",null,"After connecting to SuperDuperDB, setting up question-your-documents in Python boils down to 2 commands."),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Set up a vector-index")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.container.vector_index import VectorIndex\nfrom superduperdb.container.listener import Listener\nfrom superduperdb.ext.openai.model import OpenAIEmbedding\n\ncollection = Collection('documents')\n\ndb.add(\n    VectorIndex(\n        identifier='my-index',\n        indexing_listener=Listener(\n            model=OpenAIEmbedding(model='text-embedding-ada-002'),\n            key='txt',\n            select=collection.find(),\n        ),\n    )\n)\n")),(0,o.kt)("p",null,"In this code snippet, the model used for creating vectors is ",(0,o.kt)("inlineCode",{parentName:"p"},"OpenAIEmbedding"),". This is completely configurable.\nYou can also use:"),(0,o.kt)("ul",null,(0,o.kt)("li",{parentName:"ul"},"CohereAI API"),(0,o.kt)("li",{parentName:"ul"},"Hugging-Face ",(0,o.kt)("inlineCode",{parentName:"li"},"transformers")),(0,o.kt)("li",{parentName:"ul"},(0,o.kt)("inlineCode",{parentName:"li"},"sentence-transformers")),(0,o.kt)("li",{parentName:"ul"},"Self built models in ",(0,o.kt)("inlineCode",{parentName:"li"},"torch"))),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"Listener")," component sets up this model to listen for new data, and compute new vectors as this data comes in."),(0,o.kt)("p",null,"The ",(0,o.kt)("inlineCode",{parentName:"p"},"VectorIndex")," connects user queries with the computed vectors and the model."),(0,o.kt)("p",null,"By adding this nested component to ",(0,o.kt)("inlineCode",{parentName:"p"},"db"),", the components are activated and ready to go for vector-search."),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Add a question-answering component")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"from superduperdb.ext.openai.model import OpenAIChatCompletion\n\nchat = OpenAIChatCompletion(\n    model='gpt-3.5-turbo',\n    prompt=(\n        'Use the following content to answer this question\\n'\n        'Do not use any other information you might have learned\\n'\n        'Only base your answer on the content provided\\n'\n        '{context}\\n\\n'\n        'Here\\'s the question:\\n'\n    ),\n)\n\ndb.add(chat)\n")),(0,o.kt)("p",null,"This command creates and configures an LLM hosted on OpenAI to operate together with MongoDB.\nThe prompt can be configured to ingest the context using the ",(0,o.kt)("inlineCode",{parentName:"p"},"{context}")," format variable.\nThe results of the vector search are pasted into this format variable."),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Question your documents!")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-python"},"input = 'Explain to me the reasons for the change of strategy in the company this year.'\n\nresponse, context = db.predict(\n    'gpt-3.5-turbo',\n    input=input,\n    context=collection.like({'txt': input}, vector_index='my-index').find()\n)\n")),(0,o.kt)("p",null,"This command executes the vector-search query in the ",(0,o.kt)("inlineCode",{parentName:"p"},"context")," parameter. The results of\nthis search are added to the prompt to prime the LLM to ground its answer on the documents\nin MongoDB."))}d.isMDXComponent=!0}}]);