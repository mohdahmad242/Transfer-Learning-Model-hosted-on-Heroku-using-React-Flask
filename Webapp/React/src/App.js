import main from "./components/main"; 
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
