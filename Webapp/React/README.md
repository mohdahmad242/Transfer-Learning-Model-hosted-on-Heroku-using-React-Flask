 <br>
 <h1 align="center">Chapter Three</h1>
 <h2 align="center">Building Frontend using React.</h2>

 <p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/badge/React-17.0.1-brightgreen?logo=React">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/badge/git-2.29.2-brightgreen?logo=git">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=react-router-dom&message=5.2.0&color=brightgreen&logo=ReactRouter">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=React-Bootstrap&message=1.4.0&color=brightgreen&logo=Bootstrap">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Build&message=pass&color=brightgreen">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Heroku&message=Deployed&color=brightgreen&logo=Heroku">
    </a>
</p>
 <br>
 
## Introduction
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
> It will pop up new tab in your browser serving on `Localhost 3000`

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

### Before we start building our project let's talk about `react-router-dom` pacakage we have installed.
```
Many modern websites are actually made up of a single page, they just look like 
multiple pages because they contain components which render like separate pages. 
These are usually referred to as SPAs - single-page applications. At its core, 
what React Router does is conditionally render certain components to display 
depending on the route being used in the URL (/ for the home page, /about for the about page, etc.).

For example, we can use React Router to connect www.knit-with-scrimba.com/ 
to www.knit-with-scrimba.com/about or www.knit-with-scrimba.com/shop
```
* Although we are using only use one route for our project, it is good to learn something extra here.
  * You can read about react-router-dom here - https://reactrouter.com/web/guides/quick-start

### Our component structure 
 * In `app.js` file we define our main component `App` and we use `react-router-dom` to access differnt coponent based on route. In our case only one `\`.
 * For route `\`, we create a component `Main.js` in `component folder`. 
 * Furture we divide our `MAIN` component in two part `Prediction` and `Example`. **We can see the beuaty of react here that we can modularized our frontend project and manage them individually.**
 
## Prediction component
* These code has to be written in `component/prediction.js` file.
* We will use this component to take input for the model and also display the result.
* To send `POST` request to the API we have created in flask blog we use a package called `axios`.
* We define two state varible with null values.
* We create a Form using `react-bootstrap` and add a `onSubmit` callback to the form. 
 * `onChange`, `onSubmit`, etc are called hooks in react you can learn about them [here](https://reactjs.org/docs/hooks-intro.html).
* We define our `onSubmit` function in our `PREDICTION` component in a way given below.
 * When user click submit button onSubmit function execute which take `input review` from the state varible which changes whenerve any change occur in `Input area.`
 * This function return the result which the `result` varible in `state` and the result is diplayed.
 
## Example component
* These code has to be written in `component/example.js` file.
* This component is straight forward we add simple text which user can copy and test our model on the fly.

## Main Component
* In this component we combine the components we created in above section.\
* We first import the components and combine them in a single `div`. As written below.

## App component.
* As we discus about the `react-router-dom` before we will use it display `Main` component when user hit `\` route.



