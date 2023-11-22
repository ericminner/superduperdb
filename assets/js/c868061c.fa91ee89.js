"use strict";(self.webpackChunknewdocs=self.webpackChunknewdocs||[]).push([[2495],{76195:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>c,contentTitle:()=>s,default:()=>p,frontMatter:()=>t,metadata:()=>d,toc:()=>a});var r=i(85893),o=i(11151);const t={sidebar_position:27},s="Serializing components with SuperDuperDB",d={id:"docs/walkthrough/serialization",title:"Serializing components with SuperDuperDB",description:"When adding a component to SuperDuperDB,",source:"@site/content/docs/walkthrough/27_serialization.md",sourceDirName:"docs/walkthrough",slug:"/docs/walkthrough/serialization",permalink:"/docs/docs/walkthrough/serialization",draft:!1,unlisted:!1,editUrl:"https://github.com/SuperDuperDB/superduperdb/tree/main/docs/content/docs/walkthrough/27_serialization.md",tags:[],version:"current",sidebarPosition:27,frontMatter:{sidebar_position:27},sidebar:"tutorialSidebar",previous:{title:"Component versioning",permalink:"/docs/docs/walkthrough/component_versioning"},next:{title:"Creating complex stacks of functionality",permalink:"/docs/docs/walkthrough/creating_stacks_of_functionality"}},c={},a=[];function l(e){const n={code:"code",h1:"h1",li:"li",p:"p",pre:"pre",ul:"ul",...(0,o.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.h1,{id:"serializing-components-with-superduperdb",children:"Serializing components with SuperDuperDB"}),"\n",(0,r.jsxs)(n.p,{children:["When adding a component to ",(0,r.jsx)(n.code,{children:"SuperDuperDB"}),",\nobjects which cannot be serialized to ",(0,r.jsx)(n.code,{children:"JSON"}),"\nare serialized to ",(0,r.jsx)(n.code,{children:"bytes"})," using one of the inbuilt\nserializers:"]}),"\n",(0,r.jsxs)(n.ul,{children:["\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"pickle"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"dill"})}),"\n",(0,r.jsx)(n.li,{children:(0,r.jsx)(n.code,{children:"torch"})}),"\n"]}),"\n",(0,r.jsxs)(n.p,{children:["Users also have the choice to create their own serializer,\nby providing a pair of functions to the ",(0,r.jsx)(n.code,{children:"Component"})," descendant\n",(0,r.jsx)(n.code,{children:"superduperdb.Serializer"}),"."]}),"\n",(0,r.jsxs)(n.p,{children:["Here is an example of how to do that, with an example ",(0,r.jsx)(n.code,{children:"tensorflow.keras"})," model,\nwhich isn't yet natively supported by ",(0,r.jsx)(n.code,{children:"superduperdb"}),", but\nmay nevertheless be supported using a custom serializer:"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"from superduperdb import Serializer\n\nfrom tensorflow.keras import Sequential, load\nfrom tensorflow.keras.layers import Dense\n\nmodel = Sequential([Dense(1, input_dim=1, activation='linear')])\n\n\ndef encode(x):\n    id = uuid.uuid4()\n    x.save(f'/tmp/{id}')\n    with open(f'/tmp/{id}', 'rb') as f:\n        b = f.read()\n    os.remove(f'/tmp/{id}')\n    return b\n\n\ndef decode(x)\n    id = uuid.uuid4()\n    with open(f'/tmp/{id}', 'wb') as f:\n        f.write(x)\n    model = load(f'/tmp/{id}')\n    os.remove(f'/tmp/{id}')\n    return model\n\n\ndb.add(\n    Serializer(\n        'keras-serializer',\n        encoder=encoder,\n        decoder=decoder,\n    )\n)\n\ndb.add(\n    Model(\n        'my-keras-model',\n        object=model,\n        predict_method='predict',\n        serializer='keras-serializer',\n    )\n)\n"})})]})}function p(e={}){const{wrapper:n}={...(0,o.a)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(l,{...e})}):l(e)}},11151:(e,n,i)=>{i.d(n,{Z:()=>d,a:()=>s});var r=i(67294);const o={},t=r.createContext(o);function s(e){const n=r.useContext(t);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function d(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:s(e.components),r.createElement(t.Provider,{value:n},e.children)}}}]);