import React, { Component } from "react";
import PREDICTION from "./prediction";
import EXAMPLE from "./example";

import {
  Navbar,
  NavDropdown,
  Nav
  } from "react-bootstrap";

class Main extends Component {

  render() {
    return (
      <div>
          <Navbar collapseOnSelect expand="lg" bg="dark" variant="dark">
            <Navbar.Brand href="#home">Senti-review</Navbar.Brand>
            <Navbar.Toggle aria-controls="responsive-navbar-nav" />
            <Navbar.Collapse id="responsive-navbar-nav">
              <Nav className="mr-auto">
                <Nav.Link href="#">Jupyter Notebook</Nav.Link>
                <NavDropdown title="Tutorial Chapters" id="collasible-nav-dropdown">
                  <NavDropdown.Item href="#">Ch-1</NavDropdown.Item>
                  <NavDropdown.Item href="#">Ch-2</NavDropdown.Item>
                  <NavDropdown.Item href="#">Ch-3</NavDropdown.Item>
                </NavDropdown>
              </Nav>
              <Nav>
              <Nav.Link href="https://www.imdb.com/">Best Movies review</Nav.Link>
                <Nav.Link eventKey={2} href="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask">
                  Github
                </Nav.Link>
              </Nav>
            </Navbar.Collapse>
          </Navbar>
      
        <div style={{display:"flex",justifyContent:"center", flexDirection:"row", padding:"10px", height:"auto", width:"auto", background:"#B3E5FC"}}>
          
            <div style={{width:"50%"}}>
                <PREDICTION></PREDICTION>
            </div>
            <div style={{width:"50%"}}>
                <EXAMPLE></EXAMPLE>
            </div>
        </div>
        </div>
    );
  }
}

export default Main;
