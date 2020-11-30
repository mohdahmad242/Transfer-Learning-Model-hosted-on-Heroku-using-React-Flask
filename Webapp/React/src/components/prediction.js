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
