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
<p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/first.png">
  </kbd>
</p> 

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
[REF](https://www.freecodecamp.org/news/react-router-in-5-minutes/)

* Although we are using only use one route for our project, it is good to learn something extra here.
  * You can read about react-router-dom here - https://reactrouter.com/web/guides/quick-start

### Our component structure 
 * In `app.js` file we define our main component `App` and we use `react-router-dom` to access differnt coponent based on route. In our case only one `\`.
 <details> 
    <summary>See Code</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>App.js</i></u> fie. </summary></h3>
    
```js
import main from "./components/main"; // Importing MAIN component
import {
  BrowserRouter as Router,
  Route
} from "react-router-dom";

function App() {
  return (
    <div className="App">
      <Router>
          <Route exact path="/" component={main} />
      </Router>
    </div>
  );
}

export default App;


```
    
</details>

 <p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/app.png">
  </kbd>
</p> 

 * For route `\`, we create a component `Main.js` in `component folder`. 
 * Furture we divide our `MAIN` component in two part `Prediction` and `Example`. **We can see the beuaty of react here that we can modularized our frontend project and manage them individually.**
 
## Prediction component
* These code has to be written in `component/prediction.js` file.
<details> 
    <summary>See Code</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>App.js</i></u> fie. </summary></h3>
    
```js
import React, { Component} from "react";

import {
    Badge,
    Button,
    InputGroup,
    Form
  } from "react-bootstrap";
  
import axios from "axios";

class PREDICTION extends Component {
    constructor(props) {
        super(props);
        this.state = {
            review:null,
            result:null
        };
        this.onChange = this.onChange.bind(this)
        this.onSubmit = this.onSubmit.bind(this)
    }


    onChange(e) {
        this.setState({ [e.target.name]: e.target.value })
      }
  
      onSubmit(e) {
        this.setState({result: null})
          e.preventDefault()
          let data ={
              "review": this.state.review
          }
          axios.post(`/predict`, data).then(res => {
            console.log(res, "result");
            this.setState({result: res.data})
                });
      }

  render() {
    return (
          <div style={{padding:"50px", background:"#c4ffe6", height:"100%"}}>
            <h3 style={{margin:"auto", marginBottom:"20px"}}>Enter movie review here to predict positive or negative(min 200 Character).</h3>
            
            <InputGroup size="lg" style={{width:"100%"}}>
            <Form onSubmit={this.onSubmit} style={{width:"100%"}}>
                    <Form.Group >
                        <Form.Control 
                        style={{width:"100%"}}
                        size="lg"
                        as="textarea" 
                        placeholder="Enter movie review... " 
                        type="text"
                        name="review"
                        value={this.state.review}
                        onChange={this.onChange}
                        />
                        </Form.Group>
                        <Button variant="primary" type="submit" style={{marginBottom:"10px"}}>
                            Predict
                        </Button>
                </Form>
            </InputGroup>
            {this.state.review == null ? " " : <p><h4>Entered review - </h4> {this.state.review}"</p>}
            {this.state.result == null ? " " : <h4>Predicted value - <Badge variant="primary">{this.state.result.toUpperCase()}</Badge></h4>}
            
          </div>
    );
  }
}

export default PREDICTION;

```
    
</details>

* We will use this component to take input for the model and also display the result.
* To send `POST` request to the API we have created in flask blog we use a package called `axios`.
 <p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/axios.png">
  </kbd>
</p> 

* We define two state varible with null values.
 <p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/state.png">
  </kbd>
</p> 

* We create a Form using `react-bootstrap` and add a `onSubmit` callback to the form. 
 * `onChange`, `onSubmit`, etc are called hooks in react you can learn about them [here](https://reactjs.org/docs/hooks-intro.html).
 
  <p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/hook.png">
  </kbd>
</p> 

* We define our `onSubmit` function in our `PREDICTION` component in a way given above.
 * When user click submit button onSubmit function execute which take `input review` from the state varible which changes whenerve any change occur in `Input area.`
 * The above function return the result which will be store in the `result` varible defined in `state` and the result will be diplayed.
 
## Example component
* These code has to be written in `component/example.js` file.
<details> 
    <summary>See Code</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>components/example.js</i></u> fie. </summary></h3>
    
```js
import React, { Component} from "react";

class EXAMPLE extends Component {
  render() {
    return (
      <div style={{padding:"50px", background:"#c4ffe6", height:"100%"}}>
            <h3 style={{alignContent:"center"}}>This is Example section.</h3>
            <h4 style={{marginBottom:"0"}}>Example 1</h4>
            <p style={{margin:"0"}}>I find it very intriguing that Lee Radziwill, Jackie Kennedy's sister and the cousin of these women, would encourage the Maysles' to make "Big Edie" and 
              "Little Edie" the subject of a film. They certainly could be considered the "skeletons" in the family closet. The extra features on the DVD include several 
              contemporary fashion designers crediting some of their ideas to these oddball women. I'd say that anyone interested in fashion would find the discussion by these 
              designers fascinating. (i.e. "Are they nuts? Or am I missing something?"). 
              This movie is hard to come by. Netflix does not have it. Facets does, though.</p>
              <h4 style={{marginBottom:"0"}}>Example 2</h4>
            <p style={{margin:"0"}}>Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake. I've seen 950+ films and this is 
              truly one of the worst of them - it's awful in almost every way: editing, pacing, storyline, 'acting,' soundtrack (the film's only song - a lame country 
              tune - is played no less than four times). The film looks cheap and nasty and is boring in the extreme. Rarely have I been so happy to see the end credits
               of a film. The only thing that prevents me giving this a 1-score is Harvey Keitel - while this is far from his best performance he at least 
               seems to be making a bit of an effort. One for Keitel obsessives only.</p>
               <h4 style={{marginBottom:"0"}}>Example 3</h4>
            <p style={{margin:"0"}}>If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.Great Camp!!!</p>
            <h4 style={{marginBottom:"0"}}>Example 4</h4>
            <p style={{margin:"0"}}>I saw this movie when I was about 12 when it came out. I recall the scariest scene was the big bird eating men dangling helplessly from parachutes 
              right out of the air. The horror. The horror.As a young kid going to these cheesy B films on Saturday afternoons, I still was tired of the 
              formula for these monster type movies that usually included the hero, a beautiful woman who might be the daughter of a professor and a happy resolution 
              when the monster died in the end. I didn't care much for the romantic angle as a 12 year old and the predictable plots. I love them now for the 
              unintentional humor.But, about a year or so later, I saw Psycho when it came out and I loved that the star, Janet Leigh, was bumped off 
              early in the film. I sat up and took notice at that point. Since screenwriters are making up the story, make it up to be as scary as possible and not 
              from a well-worn formula. There are no rules.</p>
          </div>
    );
  }
}

export default EXAMPLE;


```
    
</details>

* This component is straight forward we add simple text which user can copy and test our model on the fly.

## Main Component
* In this component we combine the components we created in above section.
<details> 
    <summary>See Code</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>components/main.js</i></u> fie. </summary></h3>
    
```js
import React, { Component } from "react";
import PREDICTION from "./prediction";  // Importing PREDICTION component
import EXAMPLE from "./example";       // Importing EXAMPLE component

class Main extends Component {
  render() {
    return (
        <div style={{display:"flex",justifyContent:"center", flexDirection:"row", padding:"10px", height:"auto", width:"auto"}}>
            <div style={{width:"50%"}}>
                <PREDICTION></PREDICTION>
            </div>
            <div style={{width:"50%"}}>
                <EXAMPLE></EXAMPLE>
            </div>
        </div>
    );
  }
}

export default Main;



```
    
</details>

* We first import the components and combine them in a single `div`. As given below.
<p align="center">
    <kbd>
  <img  src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/main.png">
  </kbd>
</p> 

# Deployment on Heroku.
We expect you have GitHub account and the knowledge of how to create repository. If not, [learn here](https://guides.github.com/activities/hello-world/)
> For React you need to add any `Heroku` specific file as Heroku can autometacally build your app.

1.**Now create a repository and push all code on github repository. If you don't know how to do that, learn it [here](https://www.datacamp.com/community/tutorials/git-push-pull)**

2. **Create New App.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h1.png">
                </kbd>
        </p>
    </details>

3. **Choose App name and region.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h2.png">
                </kbd>
        </p>
    </details>

4. **Link your Github account with Heroku, search your repository where you have pushed all your code and `connect`.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h3.png">
                </kbd>
        </p>
    </details>

5. **Choose branch, `Enable automatic deploy` so that it can automatically build your app when you push any changes to your repository and hit `Deploy Branch`.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h4.png">
                </kbd>
        </p>
    </details>

6. **You can see App Build log. It will display any errors if occurs.**

    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h5.png">
                </kbd>
        </p>
    </details> 
    
7. **Finally after successful build you can launch your app by clicking `View`**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/react/h6.png">
                </kbd>
        </p>
    </details> 
    
    
***
## Summary and Conclusion
***
### Refrence
* https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
*** 

