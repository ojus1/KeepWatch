var text_utils = require('./utils');
var DecisionTreeClassifier = require('./DecisionTree');

var url = "himselp.net.in/css/acrord.exe"; //Bad url
//var url = "google.com";

var features = text_utils.bow(url);
//console.log(features.length);
//console.log(features);


var output = DecisionTreeClassifier.predict(features);
console.log("The url to test is: "+ url);
console.log("Ground Truth: " + "Bad");
console.log("Predicted output(1 = Bad, 0 = Good): " + output);