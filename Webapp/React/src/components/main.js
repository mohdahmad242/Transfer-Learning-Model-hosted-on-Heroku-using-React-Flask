import React, { Component } from "react";
import PREDICTION from "./prediction";
import EXAMPLE from "./example";

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
