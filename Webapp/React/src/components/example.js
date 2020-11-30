import React, { Component} from "react";

import {
  Badge,
  ListGroup
} from "react-bootstrap";

class EXAMPLE extends Component {

  render() {
    return (
      <div style={{padding:"50px", height:"100%"}}>
            <h3 style={{alignContent:"center"}}>Sample review to test model.</h3>
            <ListGroup>
            <ListGroup.Item variant="info">
            <h4 style={{marginBottom:"0"}}>Example 1</h4>
            <p style={{margin:"0"}}>No one expects the Star Trek movies to be high art, but the fans do expect a movie that is as good as some of the best episodes. Unfortunately, this movie had a muddled, 
            implausible plot that just left me cringing - this is by far the worst of the nine (so far) movies. Even the chance to watch the well known characters interact in another movie can't save this 
            movie - including the goofy scenes with Kirk, Spock and McCoy at Yosemite. I would say this movie is not worth a rental, 
            and hardly worth watching, however for the True Fan who needs to see all the movies, renting this movie is about the only way you'll see it - even the cable channels avoid this movie.</p>
            </ListGroup.Item>
            <ListGroup.Item variant="info">
            <h4 style={{marginBottom:"0"}}>Example 2</h4>
            <p style={{margin:"0"}}>Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake. I've seen 950+ films and this is 
              truly one of the worst of them - it's awful in almost every way: editing, pacing, storyline, 'acting,' soundtrack (the film's only song - a lame country 
              tune - is played no less than four times). The film looks cheap and nasty and is boring in the extreme. Rarely have I been so happy to see the end credits
               of a film. The only thing that prevents me giving this a 1-score is Harvey Keitel - while this is far from his best performance he at least 
               seems to be making a bit of an effort. One for Keitel obsessives only.</p>
               </ListGroup.Item>
            <ListGroup.Item variant="info">
            <h4 style={{marginBottom:"0"}}>Example 3</h4>
            <p style={{margin:"0"}}>If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.Great Camp!!!</p>
            </ListGroup.Item>
            <ListGroup.Item variant="info">
            <h4 style={{marginBottom:"0"}}>Example 4</h4>
            <p style={{margin:"0"}}>I saw this movie when I was about 12 when it came out. I recall the scariest scene was the big bird eating men dangling helplessly from parachutes 
              right out of the air. The horror. The horror.As a young kid going to these cheesy B films on Saturday afternoons, I still was tired of the 
              formula for these monster type movies that usually included the hero, a beautiful woman who might be the daughter of a professor and a happy resolution 
              when the monster died in the end. I didn't care much for the romantic angle as a 12 year old and the predictable plots. I love them now for the 
              unintentional humor.But, about a year or so later, I saw Psycho when it came out and I loved that the star, Janet Leigh, was bumped off 
              early in the film. I sat up and took notice at that point. Since screenwriters are making up the story, make it up to be as scary as possible and not 
              from a well-worn formula. There are no rules.</p>
              </ListGroup.Item>
              </ListGroup>
          </div>
    );
  }
}

export default EXAMPLE;
