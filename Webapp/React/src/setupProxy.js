const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use("/predict",
    createProxyMiddleware( { target: "https://imdbmoviereviewapp.herokuapp.com" ,changeOrigin: true})
  );
};
