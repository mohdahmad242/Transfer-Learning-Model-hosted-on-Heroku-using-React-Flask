# Introduction
In this tutorial we will build a simple React App which use the API we developed in [Flask tutorial](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Webapp/Flask/README.md).
## What is React?
React is javascript framework developed by Facebook developers. React is a declarative, efficient, and flexible JavaScript library for building user interfaces. It lets you compose complex UIs from small and isolated pieces of code called “components”[[ref]](https://reactjs.org/tutorial/tutorial.html). It uses virtual DOM (JavaScript object), which improves the performance of the app. The JavaScript virtual DOM is faster than the regular DOM. Its modeular we can write code for individual component and manage it saparetly. We can use ReactJS on the client and server-side as well as with other frameworks.

## Pre-Requisities
We assume you are familiar with these concepts - 
* HTML  [Learn here](https://www.w3schools.com/html/)
* CSS  [Learn here](https://www.w3schools.com/css/)
* JavaScript  [Learn here](https://www.w3schools.com/js/DEFAULT.asp)
 

### Getting started 
* Before we start writing we need to setup development environment for this you need to have `node>=8.10` and `npm>=6.5` installed. 
To create new project witre following code in terminal of a directory of your choice.
```bash
  npx create-react-app react-app
```
* To run this initial project `cd` in to `react-app` directory and hit `npm start`.
* Now we need to download some packages we will use in our project.
```bash
  npm install react-router-dom
  npm install react-bootstrap
  npm install axios
  npm install http-proxy-middleware
  
      OR
  
  npm install axios http-proxy-middleware react-router-dom react-bootstrap
```
> We will understand each package we install later in this tutorial.

### File System 
We need have a clean flie structure which help us to manage our code and easy to resolves bugs.
* We remove a few files - `logo.svg, App.css, App.test.js, index.css, favicon.ico, setupTests.js`.
* And added some files and folder.
    `component folder` where we store all our component files.
<table>
<tr>
<th>Initial file structure</th>
<th>Our projetc file structure</th>
</tr>
<tr>
<td>
<pre>
    react-app
    ├── README.md
    ├── node_modules
    ├── package.json
    ├── .gitignore
    ├── public
    │   ├── favicon.ico
    │   ├── index.html
    │   └── manifest.json
    └── src
        ├── App.css
        ├── App.js
        ├── App.test.js
        ├── index.css
        ├── index.js
        ├── logo.svg
        └── serviceWorker.js
        └── setupTests.js
</pre>
</td>
<td>

<pre>
    react-app
    ├── README.md
    ├── node_modules
    ├── package.json
    ├── .gitignore
    ├── public
    │   ├── index.html
    │   └── manifest.json
    └── src
        └── components  ----- (Component Folder)
        │   ├── example.js
        │   ├── main.js
        │   └── prediction.js
        ├── App.js
        ├── index.js
        ├── setupProxy.js --- To set proxy as we use external API.(Explained later in this tutorial)
        └── serviceWorker.js
</pre>
</td>
</tr>
</table>

### Let's start with main `App.js` file.
